/// tsymalla: Aggressive inter-procedural scalar replacement of aggregates
///
/// TODO: Document this!
///
/// Non-trivial cases that are handled:
///  - memset that spans multiple fields of a struct/array
///  - overlapping access ranges of different primitive types
///
/// WARNING: This code assumes a little-endian target with 8-bit bytes.

#include "llvm/Transforms/IPO/SROA.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/SSAUpdaterBulk.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>

#define DEBUG_TYPE "iposroa"

using namespace llvm;

static cl::opt<unsigned> ScalarizationScanLimit(
    "ipsroa-scalarization-load-scan-limit",
    cl::desc("Maximum amount of iterations being used to scan for the users of "
             "a load to determine if there are uses in another BB."),
    cl::value_desc("load-scan-limit"), cl::init(8));

namespace {

static Function *getContainingFunction(Value *V) {
  if (auto *Arg = dyn_cast<Argument>(V))
    return Arg->getParent();
  if (auto *Inst = dyn_cast<Instruction>(V))
    return Inst->getFunction();

  llvm_unreachable("No containing function");
}

/// A memory access info relative to some base pointer.
struct AccessType {
  /// Byte range of the access type relative to the base pointer.
  IPSROAPass::Range Range;

  /// Type of access or null for memset / memcpy
  Type *T = nullptr;

  /// Number of occurrences of this offset/size/type signature
  unsigned Count = 0;
};

/// Map of memory access kinds relative to some base pointer.
class AccessTypeMap {
public:
  void mergeAccessType(const AccessType &AT);

  ArrayRef<AccessType> accessTypes() const { return AccessTypeList; }

private:
  // Access types through the pointer, sorted by offset and size (for same
  // offset, larger sizes come before smaller sizes).
  std::vector<AccessType> AccessTypeList;
};

void AccessTypeMap::mergeAccessType(const AccessType &AT) {
  static auto Comparator = [](const AccessType &LHS, const AccessType &RHS) -> bool {
    if (LHS.Range.Offset < RHS.Range.Offset)
      return true;
    if (LHS.Range.Offset > RHS.Range.Offset)
      return false;
    if (LHS.Range.Size > RHS.Range.Size)
      return true;
    if (LHS.Range.Size < RHS.Range.Size)
      return false;

    return false;
  };

  auto B = llvm::lower_bound(AccessTypeList, AT, Comparator);
  auto E = llvm::upper_bound(AccessTypeList, AT, Comparator);

  for (AccessType &Other : llvm::make_range(B, E)) {
    assert(Other.Range.Offset == AT.Range.Offset);
    assert(Other.Range.Size == AT.Range.Size);

    if (Other.T == AT.T) {
      Other.Count += AT.Count;
      return;
    }
  }

  AccessTypeList.insert(E, AT);
}

} // anonymous namespace

unsigned IPSROAPass::BasicBlockOrderCache::get(Instruction *I) {
  if (auto It = m_map.find(I); It != m_map.end())
    return It->second;

  BasicBlock *BB = I->getParent();
  unsigned Idx = 0;
  for (Instruction &CurrInst : *BB)
    m_map[&CurrInst] = Idx++;

  return m_map.find(I)->second;
}

PreservedAnalyses IPSROAPass::run(Module &M, ModuleAnalysisManager &Analyses) {
  LLVM_DEBUG(dbgs() << "IPSROA on module: " << M.getName() << "\n");

  m_functions.clear();
  m_scalarizedFunctions.clear();
  m_ArgToScalarArgsMapping.clear();
  m_OldArgToNewArgMapping.clear();

  m_module = &M;
  m_dataLayout = &M.getDataLayout();
  m_BasicBlockOrder.reset();

  collectFunctions();
  scanAllocas();
  buildScalarizationPlans(Analyses);
  findScalarizableArguments();

  bool Changed = runScalarization(Analyses);

  if (Changed)
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}

IPSROAPass::ScalarizationAnalysis &IPSROAPass::getAnalysis(Value *V) {
  return m_analysis.find(V)->second;
}

IPSROAPass::ScalarizationPlan &IPSROAPass::getPlan(Value *V) {
  return m_plan.find(V)->second;
}

void IPSROAPass::collectFunctions() {
  LLVM_DEBUG(dbgs() << "Collecting functions to process\n");
  for (Function &Func : m_module->functions()) {
    if (Func.isDeclaration())
      continue;

    m_functions.push_back(&Func);
  }
}

bool IPSROAPass::shouldHandleAlloca(AllocaInst *Alloca) const {
  if (!Alloca->isStaticAlloca())
    return false;

  if (Alloca->isArrayAllocation()) {
    // Presumably we could handle this, but it's unlikely to be worth it.
    LLVM_DEBUG(dbgs() << "  Skipping static array alloca: " << *Alloca << '\n');

    return false;
  }

  return true;
}

/// Step 1: Analysis of allocas and arguments: find arguments that are based
/// (possibly indirectly, via multiple function calls) on allocas that have
/// no function-local obstruction to scalarization.
void IPSROAPass::scanAllocas() {
  LLVM_DEBUG(dbgs() << "First pass: Scan allocas\n");

  DenseSet<Argument *> Args;
  SmallVector<Value *, 8> Worklist;
  for (Function *Func : m_functions) {
    for (Instruction &I : Func->getEntryBlock()) {
      if (auto *Alloca = dyn_cast<AllocaInst>(&I)) {
        if (!shouldHandleAlloca(Alloca)) {
            continue;
        }
        
        Worklist.push_back(Alloca);
        do {
          Value *Current = Worklist.pop_back_val();
          ScalarizationAnalysis Analysis = analyze(Current);
          if (!Analysis.IsScalarizable)
            continue;

          for (const auto &Access : Analysis.Accesses) {
            if (Access.FuncCallArgument &&
                Args.insert(Access.FuncCallArgument).second)
              Worklist.push_back(Access.FuncCallArgument);
          }

          m_analysis.try_emplace(Current, std::move(Analysis));
          m_candidatePointers[getContainingFunction(Current)].push_back(
              Current);
        } while (!Worklist.empty());
      }
    }
  }
}

/// Step 2: Complete scalarization preplans by filling in cross-function
/// information.
void IPSROAPass::buildScalarizationPlans(ModuleAnalysisManager &Analyses) {
  m_callGraph = &Analyses.getResult<LazyCallGraphAnalysis>(*m_module);

  LLVM_DEBUG(dbgs() << "Second pass: Build plans\n");

  m_callGraph->buildRefSCCs();
  for (LazyCallGraph::RefSCC &RefSCC : m_callGraph->postorder_ref_sccs()) {
    for (LazyCallGraph::SCC &SCC : RefSCC) {
      for (LazyCallGraph::Node &Node : SCC) {
        Function &Func = Node.getFunction();
        if (Func.isDeclaration())
          continue;

        auto FuncIt = m_candidatePointers.find(&Func);
        if (FuncIt == m_candidatePointers.end())
          continue;

        for (Value *Pointer : FuncIt->second)
          preplan(Pointer);
      }
    }
  }
}

/// Step 3: Explore the call graph starting at scalarizable allocas to find
/// functions whose arguments should be scalarized.
void IPSROAPass::findScalarizableArguments() {
  DenseSet<Argument *> ArgumentsToScalarize;
  SmallVector<Argument *, 8> Worklist;
  for (Function *Func : m_functions) {
    auto FuncIt = m_candidatePointers.find(Func);
    if (FuncIt == m_candidatePointers.end())
      continue;

    for (Value *Pointer : FuncIt->second) {
      if (!isa<AllocaInst>(Pointer))
        continue;

      auto &Analysis = getAnalysis(Pointer);
      if (Analysis.IsScalarizable) {
        for (const auto &Access : Analysis.Accesses) {
          if (Access.FuncCallArgument &&
              ArgumentsToScalarize.insert(Access.FuncCallArgument).second)
            Worklist.push_back(Access.FuncCallArgument);
        }

        while (!Worklist.empty()) {
          Argument *Arg = Worklist.pop_back_val();
          auto &Analysis = getAnalysis(Arg);
          assert(Analysis.IsScalarizable);

          for (const auto &Access : Analysis.Accesses) {
            if (Access.FuncCallArgument) {
              if (ArgumentsToScalarize.insert(Access.FuncCallArgument).second)
                Worklist.push_back(Access.FuncCallArgument);
            }
          }
        }
      }
    }
  }
}

/// Step 4: Scalarize arguments and allocas while walking the call graph.
bool IPSROAPass::runScalarization(ModuleAnalysisManager &Analyses) {
  bool Changed = false;
  FunctionAnalysisManager &FAM =
      Analyses.getResult<FunctionAnalysisManagerModuleProxy>(*m_module)
          .getManager();
  LLVM_DEBUG(dbgs() << "Final pass: scalarization\n");

  for (LazyCallGraph::RefSCC &RefSCC : m_callGraph->postorder_ref_sccs()) {
    for (LazyCallGraph::SCC &SCC : RefSCC) {
      for (LazyCallGraph::Node &Node : SCC) {
        Function &Func = Node.getFunction();
        // tsymalla: Exclude compute shaders for now.
        if (Func.isDeclaration() || Func.getName().startswith("@main") || Func.getName().contains("cs"))
          continue;

        auto FuncIt = m_candidatePointers.find(&Func);
        if (FuncIt == m_candidatePointers.end())
          continue;

        // First, scalarize all the allocas
        SmallVector<Value *, 2> ArgumentsToScalarize;
        DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(Func);

        for (Value *Pointer : FuncIt->second) {
          LLVM_DEBUG(dbgs() << "Trying to scalarize " << *Pointer << "\n");
          const auto &Analysis = getAnalysis(Pointer);
          if (!Analysis.IsScalarizable) {
            LLVM_DEBUG(dbgs()
                       << "Not scalarizable: skipping " << *Pointer << "\n");
            continue;
          }

          if (auto *Alloca = dyn_cast<AllocaInst>(Pointer)) {
            const auto &Plan = getPlan(Pointer);
            SmallVector<Value *, 8> Inputs;
            for (const auto &S : Plan.Scalars)
              Inputs.push_back(UndefValue::get(S.T));

            Changed |= scalarize(DT, Alloca, Inputs, nullptr);

            Alloca->replaceAllUsesWith(UndefValue::get(Alloca->getType()));
            Alloca->eraseFromParent();
          } else {
            // tsymalla: Mark the pointer as scalarizable argument.
            ArgumentsToScalarize.push_back(Pointer);
          }
        }

        // tsymalla: Scalarize the arguments.
        if (!ArgumentsToScalarize.empty()) {
          LLVM_DEBUG(dbgs()
                     << "Scalarizing arguments of " << Func.getName() << "\n");

          // tsymalla: Collect all non-scalarizable and scalarized args and their
          // positions in the new function signature.
          m_ArgToScalarArgsMapping.clear();
          m_OldArgToNewArgMapping.clear();

          size_t CurrentArgIdx = 0;
          size_t ArgIdxWithOffset = 0;
          for (auto &FuncArg : Func.args()) {
            bool DidProcessScalarizedArgument = false;
            // tsymalla: The position of the argument in the original function and its
            // future position in the cloned function with the scalarized
            // arguments are being computed.
            // Process in reverse order, as the pointers are appended to the back.
            for (Value *ScalarizableArg : llvm::reverse(ArgumentsToScalarize)) {
              if (&FuncArg != ScalarizableArg) {
                continue;
              }

              DidProcessScalarizedArgument = true;

              // We found the scalarizable argument. Generate a set of offsetted
              // indices.
              const auto &Plan = getPlan(ScalarizableArg);
              for (size_t ScalarIdx = 0; ScalarIdx < Plan.Scalars.size();
                   ++ScalarIdx) {
                m_ArgToScalarArgsMapping[CurrentArgIdx].push_back(
                    ArgIdxWithOffset + ScalarIdx);
              }

              ArgIdxWithOffset += Plan.Scalars.size() - 1;

              break;
            }

            // tsymalla: If this argument was not processed in this iteration, then 
            // correctly compute the position in the new argument list.
            if (!DidProcessScalarizedArgument) {
              // We also map the non-scalarized arguments to its position in the
              // new functions argument list by using the original argument
              // pointer.
              //arg1 = v4, arg2 = f1, arg3 = f2
              // f1, f1, f1, f1, arg2, arg3
              LLVM_DEBUG(dbgs() << "Computing index mapping for " << FuncArg << "\n");
              m_OldArgToNewArgMapping[&FuncArg] = ArgIdxWithOffset;
            }

            ++CurrentArgIdx;
            ++ArgIdxWithOffset;
          }

          LLVM_DEBUG(dbgs() << "----------\n");
          LLVM_DEBUG(dbgs() << "Computed the following index mapping for the non-scalarized arguments: \n");
          for (auto &[Arg, Index] : m_OldArgToNewArgMapping) {
            LLVM_DEBUG(dbgs() << "Argument: " << *Arg << " Index: " << Index << "\n");
          }
          LLVM_DEBUG(dbgs() << "----------\n");

          // tsymalla: Create the new dummy placeholder function. It is empty at this 
          // point.
          Function *ClonedFunc = createDummyFunctionWithScalarizedArgs(&Func);
          m_OldToNewFuncMapping[&Func] = ClonedFunc;

          // tsymalla: Change the instructions to access the new arguments.
          while (!ArgumentsToScalarize.empty()) {
            Value *CurrentArg = ArgumentsToScalarize.pop_back_val();
            const auto &Plan = getPlan(CurrentArg);
            SmallVector<Value *, 8> Inputs;
            for (const auto &S : Plan.Scalars)
              Inputs.push_back(UndefValue::get(S.T));
            Changed |=
                scalarize(DT, CurrentArg, Inputs, nullptr, true, ClonedFunc);
          }

          // tsymalla: Then, move over the instructions to the new callee.
          while (!Func.empty()) {
            BasicBlock *BB = &Func.front();
            BB->removeFromParent();
            BB->insertInto(ClonedFunc);
          }

          ClonedFunc->takeName(&Func);
        }
      }
    }
  }

  // tsymalla: Update the callee for all call instructions to point to the new function
  // with a scalarized signature.
  for (auto &[OldFunc, CallInstrs] : m_FuncCalls) {
    for (CallInst *CI : CallInstrs) {
      CI->setCalledFunction(m_OldToNewFuncMapping[OldFunc]);
    }
  }

  return Changed;
}

/// Analyze function-local uses of \p pointer (which is expected to be a static
/// alloca or a function argument) to determine whether and how it can be
/// scalarized.
IPSROAPass::ScalarizationAnalysis IPSROAPass::analyze(Value *Pointer) {
  ScalarizationAnalysis Result;

  LLVM_DEBUG(dbgs() << "Evaluate " << getContainingFunction(Pointer)->getName()
                    << ": " << *Pointer << '\n');

  // Step 1: Build the access map.

  unsigned PointerBitSize = m_dataLayout->getTypeSizeInBits(Pointer->getType());

  // Value, Offset
  SmallVector<std::pair<Value *, int64_t>, 8> Worklist;
  Worklist.emplace_back(Pointer, 0);
  do {
    auto [CurrentValue, CurrentOffset] = Worklist.pop_back_val();

    for (Use &Use : CurrentValue->uses()) {
      User *User = Use.getUser();

      if (auto *BitCast = dyn_cast<BitCastInst>(User)) {
        Worklist.emplace_back(BitCast, CurrentOffset);
        Result.otherUsers.push_back(BitCast);
        continue;
      }

      if (auto *GEP = dyn_cast<GetElementPtrInst>(User)) {
        APInt Offset = APInt::getNullValue(PointerBitSize);
        Offset += CurrentOffset;
        if (!GEP->accumulateConstantOffset(*m_dataLayout, Offset)) {
          LLVM_DEBUG(dbgs()
                     << "  Bail due to non-constant GEP: " << *GEP << '\n');
          Result.IsScalarizable = false;
          return Result;
        }

        Worklist.emplace_back(GEP, Offset.getSExtValue());
        Result.otherUsers.push_back(GEP);
        continue;
      }

      if (isa<LoadInst>(User) || isa<StoreInst>(User)) {
        Type *CurrentType;
        bool NeedsSync;

        if (auto *Load = dyn_cast<LoadInst>(User)) {
          CurrentType = Load->getType();
          NeedsSync = Load->isAtomic() || Load->isVolatile();
        } else {
          auto *Store = cast<StoreInst>(User);
          if (Store->getValueOperand() == CurrentValue) {
            LLVM_DEBUG(dbgs() << "  Bail due to capture: " << *Store << '\n');
            Result.IsScalarizable = false;
            return Result;
          }

          CurrentType = Store->getValueOperand()->getType();
          NeedsSync = Store->isAtomic() || Store->isVolatile();
        }

        if (NeedsSync) {
          LLVM_DEBUG(dbgs() << "  Bail due to sync: " << *User << '\n');
          Result.IsScalarizable = false;
          return Result;
        }

        if (CurrentOffset < 0) {
          LLVM_DEBUG(dbgs() << "  Bail due to load/store at negative offset "
                            << CurrentOffset << ": " << *User << '\n');
          Result.IsScalarizable = false;
          return Result;
        }

        ScalarizationAnalysis::Access Access;
        Access.Instruction = cast<Instruction>(User);
        Access.Range.Offset = CurrentOffset;
        Access.Range.Size =
            m_dataLayout->getTypeStoreSize(CurrentType).getFixedSize();
        Access.T = CurrentType;
        Result.addAccess(Access);
        continue;
      }

      if (auto *CB = dyn_cast<CallBase>(User)) {
        if (auto *Intrinsic = dyn_cast<IntrinsicInst>(CB)) {
          switch (Intrinsic->getIntrinsicID()) {
          case Intrinsic::lifetime_start:
          case Intrinsic::lifetime_end:
            // Lifetime intrinsics can be ignored
            Result.otherUsers.push_back(Intrinsic);
            continue;

          case Intrinsic::memset: {
            auto *MemsetInst = cast<MemSetInst>(Intrinsic);
            if (!MemsetInst->isVolatile()) {
              if (auto *Length =
                      dyn_cast<ConstantInt>(MemsetInst->getLength())) {
                ScalarizationAnalysis::Access Access;
                Access.Instruction = Intrinsic;
                Access.Range.Offset = CurrentOffset;
                Access.Range.Size = Length->getZExtValue();
                Result.addAccess(Access);
                continue;
              }
            }
            break;
          }
          }
        } else if (Function *Callee = CB->getCalledFunction()) {
          unsigned OperandNo = Use.getOperandNo();
          if (OperandNo < Callee->arg_size()) {
            // Optimistically assume we can scalarize in the callee.
            ScalarizationAnalysis::Access Access;
            Access.Instruction = CB;
            Access.FuncCallArgument = &*(Callee->arg_begin() + OperandNo);
            Access.Range.Offset = CurrentOffset;
            Access.Range.Size = 0; // not known yet
            Result.addAccess(Access);
            continue;
          }
        }
      }

      LLVM_DEBUG(dbgs() << "  Bail due to unhandled user: " << *User << '\n');
      Result.IsScalarizable = false;
      return Result;
    }
  } while (!Worklist.empty());
  assert(Result.IsScalarizable);

  // Step 2: Resolve accesses into scalarized elements.
  LLVM_DEBUG(dbgs() << "Accesses:\n");

  for (const auto &Access : Result.Accesses) {
    LLVM_DEBUG(dbgs() << "    " << Access.Range.Offset << '+'
                      << Access.Range.Size << ": " << *Access.Instruction
                      << '\n';);
  }

  return Result;
}

void IPSROAPass::ScalarizationAnalysis::addAccess(const Access &Access) {
  size = std::max(size, Access.Range.Offset + Access.Range.Size);
  Accesses.push_back(Access);
}

/// Create the scalarization plan for the given pointer.
void IPSROAPass::preplan(Value *Pointer) {
  ScalarizationAnalysis &Analysis = getAnalysis(Pointer);
  assert(Analysis.IsScalarizable);

  LLVM_DEBUG(dbgs() << "Preplan for "
                    << getContainingFunction(Pointer)->getName() << ": "
                    << *Pointer << '\n');

  // Step 1: Collect all access types.
  AccessTypeMap TypedAccessMap;
  AccessTypeMap UntypedAccessMap;

  for (const auto &Access : Analysis.Accesses) {
    if (!Access.FuncCallArgument) {
      AccessType AT;
      AT.Range = Access.Range;
      AT.T = Access.T;
      AT.Count = 1;

      if (AT.T)
        TypedAccessMap.mergeAccessType(AT);
      else
        UntypedAccessMap.mergeAccessType(AT);
    } else {
      auto PlanIt = m_plan.find(Access.FuncCallArgument);
      if (PlanIt == m_plan.end()) {
        // This can happen due to recursion or due to the callee
        // being unscalarizable.
        LLVM_DEBUG(dbgs() << "    Bail due to unhandled call: "
                          << *Access.Instruction << '\n');
        Analysis.IsScalarizable = false;
        return;
      }

      for (const auto &Scalar : PlanIt->second.Scalars) {
        AccessType AT;
        AT.Range.Offset = Access.Range.Offset + Scalar.Range.Offset;
        AT.Range.Size = Scalar.Range.Size;
        AT.T = Scalar.T;
        AT.Count = 1;

        if (AT.T)
          TypedAccessMap.mergeAccessType(AT);
        else
          UntypedAccessMap.mergeAccessType(AT);
      }
    }
  }

  // Step 2: Scan typed accesses and collect reasonably-looking scalars.
  ArrayRef<AccessType> AccessTypes = TypedAccessMap.accessTypes();
  std::vector<Scalar> Scalars;
  unsigned TypedEnd = 0;
  unsigned Idx = 0;
  while (Idx < AccessTypes.size()) {
    AccessType AT = AccessTypes[Idx++];
    Type *CheckT = AT.T;

    if (CheckT) {
      if (auto *VecTy = dyn_cast<FixedVectorType>(CheckT)) {
        CheckT = VecTy->getElementType();
      }

      if (auto *IntTy = dyn_cast<IntegerType>(CheckT)) {
        if ((IntTy->getBitWidth() % 8) != 0) {
          LLVM_DEBUG(dbgs() << "  Bail due to type not a multiple of 8 bits: "
                            << *IntTy << '\n');
          Analysis.IsScalarizable = false;
          return;
        }
      } else if (!CheckT->isFloatingPointTy()) {
        // TODO: We really should support pointer types here, but we'll have
        //       to be careful to avoid aliasing and e.g. problems related to
        //       pointer provenance.
        LLVM_DEBUG(dbgs() << "  Bail due to unsupported type: " << *CheckT
                          << '\n');
        Analysis.IsScalarizable = false;
        return;
      }
    }

    if (AT.Range.Offset < TypedEnd) {
      if (AT.Range.Offset + AT.Range.Size <= TypedEnd) {
        // Simply ignore this access.
        continue;
      }

      LLVM_DEBUG(dbgs() << "  Access across current end (" << TypedEnd
                        << "): " << AT.Range.Offset << '+'
                        << AT.Range.Size << ": " << *AT.T
                        << '\n');

      // Record an untyped access to ensure we fill any resulting gaps.
      unsigned Overlap = TypedEnd - AT.Range.Offset;
      AT.Range.processSize(Overlap);
      UntypedAccessMap.mergeAccessType(AT);
      continue;
    }

    // See if we can find a better type for the current range.
    while (Idx < AccessTypes.size()) {
      const auto &Other = AccessTypes[Idx];

      if (Other.Range.Offset != AT.Range.Offset ||
          Other.Range.Size != AT.Range.Size)
        break;

      if (!AT.T || (Other.T && Other.Count > AT.Count)) {
        AT.T = Other.T;
        AT.Count = Other.Count;
      }

      ++Idx;
    }

    // Emit the scalar.
    Scalar S;

    Type *T = AT.T;
    unsigned Copies = 1;
    if (auto *VecTy = dyn_cast<FixedVectorType>(T)) {
      // TODO: Should we really scalarize vectors? I think it depends on
      //       the use case. In the particular use case I am looking at
      //       as I write this, vector-typed accesses have overlaps with
      //       differently-sized accesses in ways that really suggest
      //       scalarization, but surely this isn't universal.
      Copies = VecTy->getNumElements();
      T = VecTy->getElementType();
    }

    S.T = T;
    S.Range.Offset = AT.Range.Offset;
    S.Range.Size = m_dataLayout->getTypeAllocSize(T);

    for (unsigned Idx = 0; Idx < Copies; ++Idx) {
      Scalars.push_back(S);
      S.Range.step();
    }

    TypedEnd = S.Range.Offset;
    assert(TypedEnd == AT.Range.Offset + AT.Range.Size);
  }

  // Step 3: Fill in any untyped gaps.
  std::vector<Scalar> UntypedGaps;
  unsigned UntypedEnd = 0;
  unsigned ScalarIdx = 0;
  for (const auto &Access : UntypedAccessMap.accessTypes()) {
    Range R = Access.Range;

    if (R.Offset < UntypedEnd) {
      unsigned Overlap = UntypedEnd - R.Offset;
      if (Overlap >= R.Size)
        continue;

      R.processSize(Overlap);
    }

    // Break the untyped range at scalars and note gaps.
    for (;;) {
      while (ScalarIdx < Scalars.size() &&
             Scalars[ScalarIdx].Range.Offset <= R.Offset) {
        const Scalar &S = Scalars[ScalarIdx++];
        if (S.Range.peekStepVal() <= R.Offset)
          continue;

        unsigned Overlap = S.Range.Offset + S.Range.Size - R.Offset;
        if (Overlap >= R.Size) {
          R.Size = 0;
          UntypedEnd = R.Offset + Overlap;
          break;
        }

        R.processSize(Overlap);
      }

      if (!R.Size)
        break;

      Scalar GapScalar;
      GapScalar.Range.Offset = R.Offset;

      if (ScalarIdx < Scalars.size() &&
          Scalars[ScalarIdx].Range.Offset < R.Offset + R.Size) {
        const Scalar &S = Scalars[ScalarIdx++];
        GapScalar.Range.Size = S.Range.Offset - R.Offset;
        UntypedEnd = S.Range.Offset + S.Range.Size;
      } else {
        GapScalar.Range.Size = R.Size;
        UntypedEnd = R.Offset + R.Size;
      }

      if (!UntypedGaps.empty() && UntypedGaps.back().Range.Offset + UntypedGaps.back().Range.Size ==
                               GapScalar.Range.Offset) {
        UntypedGaps.back().Range.Size += GapScalar.Range.Size;
      } else {
        UntypedGaps.push_back(GapScalar);
      }

      if (R.Offset + R.Size <= UntypedEnd)
        break;

      R.Size = R.Offset + R.Size - UntypedEnd;
      R.Offset = UntypedEnd;
    }
  }

  std::vector<Scalar> TypedGaps;
  for (Scalar &S : UntypedGaps) {
    if (S.T) {
      TypedGaps.push_back(S);
      continue;
    }

    // We have an untyped gap. Invent a type for it.
    Range R = S.Range;

    // First, align to multiples of 8 bytes.
    for (unsigned ElementSize = 1; ElementSize <= 8; ElementSize *= 2) {
      if (R.Size < ElementSize)
        break;
      if ((R.Offset & (ElementSize - 1)) == 0)
        continue;

      Scalar TypedScalar;
      TypedScalar.Range.Offset = R.Offset;
      TypedScalar.Range.Size = ElementSize;
      TypedScalar.T = IntegerType::get(Pointer->getContext(), ElementSize * 8);
      TypedGaps.push_back(TypedScalar);

      R.processSize(ElementSize);
    }

    // Now fill the bulk.
    for (unsigned ElementSize = 8; ElementSize >= 1; ElementSize /= 2) {
      if (R.Size < ElementSize)
        continue;

      Scalar TypedScalar;
      TypedScalar.Range.Offset = R.Offset;
      TypedScalar.Range.Size = ElementSize;
      TypedScalar.T = IntegerType::get(Pointer->getContext(), ElementSize * 8);

      if (R.Size >= 2 * ElementSize) {
        unsigned Count = R.Size / ElementSize;
        TypedScalar.T = FixedVectorType::get(TypedScalar.T, Count);
        TypedScalar.Range.Size *= Count;
      }

      TypedGaps.push_back(TypedScalar);

      R.processSize(TypedScalar.Range.Size);
    }

    assert(R.Size == 0);
  }

  // Step 4: Save the plan.
  ScalarizationPlan &Plan = m_plan[Pointer];
  std::merge(Scalars.begin(), Scalars.end(), TypedGaps.begin(), TypedGaps.end(),
             std::back_inserter(Plan.Scalars));

  LLVM_DEBUG(dbgs() << "  Plan:\n");
  llvm::for_each(Plan.Scalars, [](const Scalar &S) {
    LLVM_DEBUG(dbgs() << "    " << S.Range.Offset << '+' << S.Range.Size << ": "
                      << *S.T << '\n');
  });
}

/// Extract bits from \p src starting at \p srcOffset, and return them in an
/// integer type of \p dstSize bits starting at \p dstOffset, with all other
/// bits zero'd out.
///
/// If \p UsePoisonGuard is true, the function guarantees that the bits of the
/// output that should be zero really are zero. A freeze instructions is
/// inserted if necessary.
///
/// \return the value and the number of extracted bits.
static std::tuple<Value *, unsigned>
createExtractBits(IRBuilder<> &Builder, Value *Src, unsigned SrcOffset,
                  unsigned DstOffset, unsigned DstSize, bool UsePoisonGuard) {
  unsigned SrcSize = Src->getType()->getPrimitiveSizeInBits();
  if (!Src->getType()->isIntegerTy())
    Src = Builder.CreateBitCast(Src, Builder.getIntNTy(SrcSize));

  unsigned HaveBits = SrcSize - SrcOffset;
  unsigned UseBits = std::min(HaveBits, DstSize - DstOffset);

  if (UseBits == DstSize) {
    // Common case where we're loading a smaller part
    // of a single scalar.
    if (SrcOffset > 0) {
      Src = Builder.CreateLShr(Src, SrcOffset);
      Src = Builder.CreateTrunc(Src, Builder.getIntNTy(UseBits));
    }

    return {Src, UseBits};
  }

  Src = Builder.CreateFreeze(Src);

  // General case. First, either zero-extend or opportunistically truncate to
  // the destination type.
  const unsigned OriginalSize = SrcSize;

  if (DstSize < SrcSize && SrcOffset + UseBits < DstSize) {
    Src = Builder.CreateTrunc(Src, Builder.getIntNTy(DstSize));
    SrcSize = DstSize;
  } else if (SrcSize < DstSize) {
    Src = Builder.CreateZExt(Src, Builder.getIntNTy(DstSize));
    SrcSize = DstSize;
  }

  // Clear out all unused bits and shift the bits we are interested in to
  // begin at the LSB
  if (SrcSize + UseBits >= std::min(SrcSize, OriginalSize)) {
    // Either there were no unused high bits initially, or they were
    // truncated away above. Only shift towards LSB
    if (SrcOffset > 0)
      Src = Builder.CreateLShr(Src, SrcOffset);
  } else {
    // There are high bits to clear out; to that first, then shift to LSB.
    Src = Builder.CreateShl(Src, SrcSize - (SrcOffset + UseBits));
    Src = Builder.CreateLShr(Src, SrcSize - UseBits);
  }

  // Shift into place.
  if (DstOffset > 0)
    Src = Builder.CreateShl(Src, DstOffset);

  // Truncate to the destination size if still required.
  if (DstSize < SrcSize)
    Src = Builder.CreateTrunc(Src, Builder.getIntNTy(DstSize));

  return {Src, UseBits};
}

/// Scalarize one alloca or function argument. This function:
///
///  - Removes old access instructions that were based on \p pointer and
///  replaces
///    them by new ones.
///  - Uses the values passed into \p inputs as the initial values of the
///    scalars in the pointer's scalarization plan.
///  - Removes other direct and indirect users of \p pointer.
///  - Transforms accesses to use the scalarized arguments of the cloned
///  function
///    instead of the memory access instructions.
///  - Does not remove \p pointer itself.
///  - If \p outputs is non-null, it is filled with a vector of each scalar's
///    output value for each return instruction in the function. If a scalar
///    isn't written to on the path to a return instruction, the corresponding
///    value is equal to the scalar's input placeholder.
///
bool IPSROAPass::scalarize(
    DominatorTree &DT, Value *Pointer, ArrayRef<Value *> Inputs,
    DenseMap<ReturnInst *, std::vector<Value *>> *Outputs, bool isArgument,
    Function *ClonedFunc) {
  assert(m_analysis.count(Pointer));
  assert(m_plan.count(Pointer));
  const ScalarizationAnalysis &Analysis = getAnalysis(Pointer);
  const ScalarizationPlan &Plan = getPlan(Pointer);
  SmallVector<Value *, 8> SSACopies;
  assert(Analysis.IsScalarizable);

  struct BbInfo {
    std::vector<const ScalarizationAnalysis::Access *> Accesses;
  };

  LLVM_DEBUG(dbgs() << "Scalarize " << getContainingFunction(Pointer)->getName()
                    << ": " << *Pointer << '\n');

  // Prepare the SSA bulk updater and insert placeholders for the scalars:
  //  * input placeholders will be passed to our caller, who is responsible
  //    for replacing them with the true initial value
  //  * dummies will be used for the initial value in each basic block; their
  //    uses will be eliminated entirely by the bulk updater
  BasicBlock *EntryBlock = &getContainingFunction(Pointer)->getEntryBlock();
  IRBuilder<> Builder(EntryBlock->getFirstNonPHIOrDbgOrLifetime());
  SSAUpdaterBulk BulkUpdater;
  std::vector<Instruction *> DummyInstructions;
  std::vector<std::string> DummyNames;

  for (unsigned Idx = 0; Idx < Plan.Scalars.size(); ++Idx) {
    DummyNames.push_back(
        (Twine(Pointer->getName()) + "." + std::to_string(Idx)).str());
  }

  for (unsigned Idx = 0; Idx < Plan.Scalars.size(); ++Idx) {
    const auto &S = Plan.Scalars[Idx];
    BulkUpdater.AddVariable(DummyNames[Idx], S.T);

    Value *Tmp = Constant::getNullValue(S.T->getPointerTo(0));
    DummyInstructions.push_back(
        Builder.CreateLoad(S.T, Tmp, Twine(DummyNames[Idx]) + ".dummy"));
  }

  // Collect accesses into basic blocks, where we will sort them according to
  // basic block order.
  //
  // Ensure the presence of the entry block, so that input placeholders get
  // added to the bulk updater even if the corresponding scalars never appear
  // in the entry block.
  DenseMap<BasicBlock *, BbInfo> BBInfos;
  BBInfos.try_emplace(&getContainingFunction(Pointer)->getEntryBlock());

  for (const auto &Access : Analysis.Accesses) {
    BasicBlock *BB = Access.Instruction->getParent();
    BBInfos[BB].Accesses.push_back(&Access);
  }

  // Rewrite basic blocks, using the dummy or input placeholder for initial
  // values.
  std::vector<Value *> PlaceholderList;
  DenseMap<size_t, bool> PlaceholderNonOverrides;
  for (auto &BBNode : BBInfos) {
    BbInfo &BBInfo = BBNode.second;
    llvm::sort(BBInfo.Accesses, [this](const auto *LHS, const auto *RHS) {
      return m_BasicBlockOrder.get(LHS->Instruction) <
             m_BasicBlockOrder.get(RHS->Instruction);
    });

    PlaceholderList.clear();
    if (BBNode.first == EntryBlock)
      PlaceholderList.insert(PlaceholderList.begin(), Inputs.begin(),
                             Inputs.end());
    else
      PlaceholderList.insert(PlaceholderList.begin(), DummyInstructions.begin(),
                             DummyInstructions.end());

    // In case this is an argument which is used to load the return value,
    // replace it with the already scalarized values.
    bool CheckReturnValue = false;
    size_t ScalarizedArgIndex = 0;
    for (const auto *Access : BBInfo.Accesses) {
      Builder.SetInsertPoint(Access->Instruction);

      // tsymalla: TODO: Get this to work. Size is not always known.
      /*size_t ScalarizedArgIndex =
          Access->Range.Offset > 0 && Access->Range.Size > 0
              ? Access->Range.Offset / Access->Range.Size
              : 0;*/

      // tsymalla: Handle arguments specifically.
      if (isArgument) {
        // tsymalla: Check if this instruction is a load whose value is used as return value.
        // In this case, the pass should use the final insertelement instruction instead.
        if (auto *Load = dyn_cast<LoadInst>(Access->Instruction)) {
          for (User *U : Load->users()) {
            if (auto *RI = dyn_cast<ReturnInst>(U)) {
              CheckReturnValue = true;
              break;
            }
          }
        }
      }

      if (isArgument && !CheckReturnValue) {
        assert(ClonedFunc &&
               "Requires a function with the scalarized arguments.");

        // tsymalla: All uses of non-scalarized arguments should map to the corresponding
        // argument in the new function.
        for (auto &ArgMapTuple : m_OldArgToNewArgMapping) {
          Argument *NewArg = ClonedFunc->getArg(ArgMapTuple.second);
          LLVM_DEBUG(dbgs() << "Replacing all uses of old arg\n" << *ArgMapTuple.first << " with\n" << *NewArg << "!\n");
          ArgMapTuple.first->replaceAllUsesWith(NewArg);
        }

        // tsymalla: The placeholders should be updated to point to the now-scalar args.
        size_t Idx = 0;
        for (auto &[Arg, Scalars] : m_ArgToScalarArgsMapping) {
          for (size_t ScalarIdx : Scalars) {
            // Don't overwrite the placeholders if they were already overridden.
            if (PlaceholderNonOverrides.find(ScalarIdx) ==
                PlaceholderNonOverrides.end() || !PlaceholderNonOverrides[ScalarIdx]) {
              //PlaceholderList[Idx] = ClonedFunc->getArg(
                  //std::min(Arg + ScalarIdx, ClonedFunc->arg_size() - 1));
              ++Idx;
            }
          }
        }
      }

      if (auto *Load = dyn_cast<LoadInst>(Access->Instruction)) {
        if (isArgument && !CheckReturnValue) {
          // tsymalla: Let every load user use either the new scalar argument or its 
          // last use.
          for (User *U : Load->users()) {
            if (U != Load->user_back()) {
              for (Use &Use : Load->uses()) {
                BulkUpdater.AddUse(ScalarizedArgIndex, &Use);
              }
            } else {
              // tsymalla: Update the available value if the consecutive loads should use the
              // result of another instruction instead, e.g. if the load is only used
              // as argument for another call.
              if (!Load->getType()->isVectorTy()) {
                LLVM_DEBUG(dbgs()
                           << "Updating PlaceholderList to use the last use of "
                           << *Load << " for rewriting: using \n"
                           << *U << " instead of \n"
                           << *PlaceholderList[ScalarizedArgIndex] << "!\n");
                PlaceholderList[ScalarizedArgIndex] = U;
                PlaceholderNonOverrides[ScalarizedArgIndex] = true;
                Load->replaceAllUsesWith(
                    ClonedFunc->getArg(ScalarizedArgIndex));
                break;
              }
            }
          }

          // Scan through the old argument list to find the argument the
          // load is using as base address.
          // Then use this argument to find the correct offset in the new argument list.
          // TODO: The load needs to be aware of the variable index within the
          // SSAUpdater. An access instruction loads data from a pointer. There
          // is a variable for each scalar. Thus, we need to know which
          // component of the vector the load accesses.


          ++ScalarizedArgIndex;
          //continue;
        }

        auto ScalarIt =
            llvm::lower_bound(Plan.Scalars, Access->Range.Offset,
                              [](const auto &LHS, unsigned RHS) {
                                return LHS.Range.Offset + LHS.Range.Size <= RHS;
                              });

        assert(ScalarIt != Plan.Scalars.end());
        assert(ScalarIt->Range.Offset <= Access->Range.Offset);

        Value *LoadValue;

        // Check for the simple case that needs no bit twiddling
        // other than potentially a bitcast (e.g. float <-> int).
        if (ScalarIt->Range.Offset == Access->Range.Offset &&
            ScalarIt->Range.Size == Access->Range.Size) {
          unsigned ScalarIdx = std::distance(Plan.Scalars.begin(), ScalarIt);

          LoadValue = PlaceholderList[ScalarIdx];
          if (LoadValue == DummyInstructions[ScalarIdx]) {
            // Uses will be replaced later by SSAUpdaterBulk, and
            // that update is sensitive to the containing basic block.
            // If there are potentially uses in a different basic block,
            // create an ssa_copy as a placeholder.

            unsigned ScanCount = 0;
            bool PossibleUseInDifferentBB = false;
            for (User *User : Load->users()) {
              Instruction *I = cast<Instruction>(User);
              if (I->getParent() != BBNode.first ||
                  ++ScanCount >= ScalarizationScanLimit) {
                PossibleUseInDifferentBB = true;
                break;
              }
            }

            if (PossibleUseInDifferentBB) {
              LoadValue =
                  Builder.CreateUnaryIntrinsic(Intrinsic::ssa_copy, LoadValue);
              SSACopies.push_back(LoadValue);
            }
          }

          if (LoadValue->getType() != Access->T)
            LoadValue = Builder.CreateBitCast(LoadValue, Access->T);
        } else {
          // Complex case: Assemble the loaded value from one or more
          // scalars at a byte level.
          //
          // NOTE: This code assumes a little-endian target.
          unsigned NumComponents = 1;
          Type *ComponentType = Access->T;

          if (auto *VecTy = dyn_cast<FixedVectorType>(ComponentType)) {
            NumComponents = VecTy->getNumElements();
            ComponentType = VecTy->getElementType();
            LoadValue = UndefValue::get(Access->T);
          }
          assert((Access->Range.Size % NumComponents) == 0);

          unsigned ComponentBytes = Access->Range.Size / NumComponents;
          unsigned ScalarIdx = std::distance(Plan.Scalars.begin(), ScalarIt);
          Value *Scalar = PlaceholderList[ScalarIdx];
          unsigned ScalarBytes = Plan.Scalars[ScalarIdx].Range.Size;
          unsigned ScalarOffset =
              Access->Range.Offset - Plan.Scalars[ScalarIdx].Range.Offset;

          if (!Scalar->getType()->isIntegerTy())
            Scalar = Builder.CreateBitCast(Scalar,
                                           Builder.getIntNTy(8 * ScalarBytes));

          for (unsigned Idx = 0; Idx < NumComponents; ++Idx) {
            Value *CurrentValue = nullptr;
            unsigned CurrentBytes = 0;

            do {
              auto [Tmp, TmpBits] = createExtractBits(
                  Builder, Scalar, 8 * ScalarOffset, 8 * CurrentBytes,
                  8 * ComponentBytes, false);
              assert(TmpBits % 8 == 0);

              if (!CurrentValue)
                CurrentValue = Tmp;
              else
                CurrentValue = Builder.CreateOr(CurrentValue, Tmp);
              CurrentBytes += TmpBits / 8;
              ScalarOffset += TmpBits / 8;

              if (ScalarOffset >= ScalarBytes) {
                ++ScalarIdx;
                if (ScalarIdx < Plan.Scalars.size()) {
                  Scalar = PlaceholderList[ScalarIdx];
                  ScalarBytes = Plan.Scalars[ScalarIdx].Range.Size;
                  ScalarOffset = 0;
                } else {
                  Scalar = nullptr;
                }
              }
            } while (CurrentBytes < ComponentBytes);
            assert(CurrentBytes == ComponentBytes);

            if (CurrentValue->getType() != ComponentType)
              CurrentValue = Builder.CreateBitCast(CurrentValue, ComponentType);

            if (isa<VectorType>(Access->T))
              LoadValue =
                  Builder.CreateInsertElement(LoadValue, CurrentValue, Idx);
            else
              LoadValue = CurrentValue;
          }
        }

        Load->replaceAllUsesWith(LoadValue);
        Load->eraseFromParent();
        continue;
      }

      if (auto *Store = dyn_cast<StoreInst>(Access->Instruction)) {
        auto ScalarIt =
            llvm::lower_bound(Plan.Scalars, Access->Range.Offset,
                              [](const Scalar &LHS, unsigned RHS) {
                                return LHS.Range.Offset + LHS.Range.Size <= RHS;
                              });
        assert(ScalarIt != Plan.Scalars.end());
        assert(ScalarIt->Range.Offset <= Access->Range.Offset);
        unsigned ScalarIdx = std::distance(Plan.Scalars.begin(), ScalarIt);

        Value *StoreValue = Store->getValueOperand();

        // tsymalla: If the pointer is an argument, make sure the scalarized value 
        // is used.
        if (isArgument && StoreValue != DummyInstructions[ScalarIdx])
          StoreValue = PlaceholderList[ScalarIdx];

        if (ScalarIt->Range.Offset == Access->Range.Offset &&
            ScalarIt->Range.Size == Access->Range.Size) {
          // Simple case where we overwrite exactly one scalar.

          if (Access->T != ScalarIt->T)
            StoreValue = Builder.CreateBitCast(StoreValue, ScalarIt->T);

          PlaceholderList[ScalarIdx] = StoreValue;
        } else {
          // General case where we may overwrite multiple scalars, and do
          // so only partially.
          unsigned NumComponents = 1;
          unsigned ComponentBytes = Access->Range.Size;

          if (auto *VectorTy = dyn_cast<FixedVectorType>(Access->T)) {
            NumComponents = VectorTy->getNumElements();
            ComponentBytes /= NumComponents;
          }

          unsigned ScalarIdx = std::distance(Plan.Scalars.begin(), ScalarIt);
          unsigned ScalarOffset =
              Access->Range.Offset - Plan.Scalars[ScalarIdx].Range.Offset;

          for (unsigned Idx = 0; Idx < NumComponents; ++Idx) {
            Value *StoreComponent = StoreValue;
            if (isa<VectorType>(Access->T))
              StoreComponent = Builder.CreateExtractElement(StoreValue, Idx);

            if (!StoreComponent->getType()->isIntegerTy()) {
              StoreComponent = Builder.CreateBitCast(
                  StoreComponent, Builder.getIntNTy(8 * ComponentBytes));
            }

            unsigned StoreOffset = 0;
            do {
              unsigned ScalarSize = Plan.Scalars[ScalarIdx].Range.Size;
              Type *ScalarType = Plan.Scalars[ScalarIdx].T;

              auto [StorePart, StoreBits] =
                  createExtractBits(Builder, StoreComponent, 8 * StoreOffset,
                                    8 * ScalarOffset, 8 * ScalarSize, true);
              assert(StoreBits % 8 == 0);

              if (StoreBits == 8 * ScalarSize) {
                // Simple case: We overwrite a scalar entirely.
                if (StorePart->getType() != ScalarType)
                  StorePart = Builder.CreateBitCast(StorePart, ScalarType);
                PlaceholderList[ScalarIdx] = StorePart;
              } else {
                // Complex case: Need to merge with the old value.
                // Freeze the old value to avoid spreading poison
                // relative to the original, non-scalarized program.
                APInt OldMask = APInt::getNullValue(8 * ScalarSize);
                if (ScalarOffset > 0)
                  OldMask |=
                      APInt::getLowBitsSet(8 * ScalarSize, 8 * ScalarOffset);
                unsigned ScalarStoreEnd = ScalarOffset + StoreBits / 8;
                if (ScalarStoreEnd < ScalarSize)
                  OldMask |= APInt::getHighBitsSet(
                      8 * ScalarSize, 8 * (ScalarSize - ScalarStoreEnd));

                Value *Tmp = PlaceholderList[ScalarIdx];
                Tmp = Builder.CreateFreeze(Tmp);
                Tmp = Builder.CreateAnd(Tmp, OldMask);
                Tmp = Builder.CreateOr(Tmp, StorePart);
                PlaceholderList[ScalarIdx] = Tmp;
              }

              StoreOffset += StoreBits / 8;
              ScalarOffset += StoreBits / 8;
              if (ScalarOffset >= ScalarSize) {
                assert(ScalarOffset == ScalarSize);
                ++ScalarIdx;
                ScalarOffset = 0;
              }
            } while (StoreOffset < ComponentBytes);
            assert(StoreOffset == ComponentBytes);
          }
        }

        Store->eraseFromParent();
        continue;
      }

      if (auto *Call = dyn_cast<CallBase>(Access->Instruction)) {
        if (auto *Intrinsic = dyn_cast<IntrinsicInst>(Call)) {
          switch (Intrinsic->getIntrinsicID()) {
          case Intrinsic::memset:
            // todo();
            break;
          default:
            llvm_unreachable("unhandled access intrinsic");
          }
        } else {
          // tsymalla: If the access is a call instruction, 
          // replace it with a new one using the scalarized arguments.
          LLVM_DEBUG(dbgs() << "Handling call: " << *Call << "\n");

          // Create a new call that uses the scalarized arguments.
          SmallVector<Value *, 1> Args;
          SmallVector<Type *, 1> ArgTys;
          bool WillCreateNewCall = false;

          DenseMap<size_t, std::vector<size_t>> ArgScalarMap;

          size_t ArgIdx = 0;
          // tsymalla: We will now create a new function call, where the arguments will
          // be a mix between the values generated from the destructurized
          // pointers and the existing arguments.
          // TODO: This incrementally recreates the call because it uses the
          // dummy instructions being generated for this specific argument. It
          // would be better to re-create the call with all arguments at once.
          for (Use &U : Call->args()) {
            if (U == Pointer) {
              for (unsigned Idx = 0; Idx < Plan.Scalars.size(); ++Idx) {
                Args.push_back(DummyInstructions[Idx]);
                ArgTys.push_back(Plan.Scalars[Idx].T);

                ArgScalarMap[ArgIdx].push_back(Idx);

                ++ArgIdx;
              }

              WillCreateNewCall = !Plan.Scalars.empty();
            } else {
              // Include default arguments
              Args.push_back(U);
              ArgTys.push_back(U->getType());

              ++ArgIdx;
            }
          }

          // tsymalla. Check if the call actually has changed.
          if (WillCreateNewCall) {
            FunctionType *CalleeTy =
                FunctionType::get(Call->getType(), ArgTys, false);
            Builder.SetInsertPoint(Call);
            // We refer to the old function here, as the new function is not
            // neccessarily created yet. It gets updated at the end of the
            // scalarization process.
            CallInst *NewCall =
                Builder.CreateCall(CalleeTy, Call->getCalledFunction(), Args);
            NewCall->setCallingConv(Call->getCallingConv());

            // Update the SSA bulk updater so the new scalar arguments are
            // made uses of the scalar variables inserted at the beginning.
            // This will make the bulk updater rewrite the arguments to
            // use the actually scalarized data instead.
            for (auto &[ArgIdx, MappedScalars] : ArgScalarMap) {
              Use *ArgUse = &NewCall->getArgOperandUse(ArgIdx);
              assert(ArgUse && "Argument not available!");
              for (std::size_t ScalarIdx : MappedScalars) {
                BulkUpdater.AddUse(ScalarIdx, ArgUse);
              }
            }

            Call->replaceAllUsesWith(NewCall);

            // The old function calls need to be remapped later.
            m_FuncCalls[Call->getCalledFunction()].push_back(NewCall);
            Call->eraseFromParent();
          }

          continue;
        }
      }

      LLVM_DEBUG(dbgs() << "Could not handle instruction: "
                        << *Access->Instruction << "\n");
      llvm_unreachable("unhandled access instruction");
    }

    // Add definitions at the end of the basic block.
    for (unsigned Idx = 0; Idx < PlaceholderList.size(); ++Idx) {
      if (PlaceholderList[Idx] != DummyInstructions[Idx]) {
        BulkUpdater.AddAvailableValue(Idx, BBNode.first, PlaceholderList[Idx]);
      }
    }
  }

  // If outputs are requested, add output markers with uses to be rewritten
  // by the bulk updater.
  if (Outputs) {
    for (BasicBlock &Bb : *EntryBlock->getParent()) {
      if (auto *Ret = dyn_cast<ReturnInst>(Bb.getTerminator())) {
        std::vector<Value *> OutputPlaceholders;

        Builder.SetInsertPoint(Ret);
        for (unsigned Idx = 0; Idx < DummyInstructions.size(); ++Idx) {
          OutputPlaceholders.push_back(Builder.CreateUnaryIntrinsic(
              Intrinsic::ssa_copy, DummyInstructions[Idx]));
        }

        Outputs->try_emplace(Ret, std::move(OutputPlaceholders));
      }
    }
  }

  // Apply bulk updates.
  for (unsigned Idx = 0; Idx < DummyInstructions.size(); ++Idx) {
    for (Use &Use : DummyInstructions[Idx]->uses())
      BulkUpdater.AddUse(Idx, &Use);
  }

  BulkUpdater.RewriteAllUses(&DT);

  // Cleanup SSA copies.
  for (Value *Value : SSACopies) {
    auto *Intrinsic = cast<IntrinsicInst>(Value);
    assert(Intrinsic->getIntrinsicID() == Intrinsic::ssa_copy);
    Intrinsic->replaceAllUsesWith(Intrinsic->getArgOperand(0));
    Intrinsic->eraseFromParent();
  }

  // Cleanup dummies.
  for (unsigned Idx = 0; Idx < DummyInstructions.size(); ++Idx)
    DummyInstructions[Idx]->eraseFromParent();

  // Cleanup outputs.
  if (Outputs) {
    for (auto &RetOutputs : *Outputs) {
      for (auto &Output : RetOutputs.second) {
        auto *Placeholder = cast<IntrinsicInst>(Output);
        assert(Placeholder->getIntrinsicID() == Intrinsic::ssa_copy);
        Output = Placeholder->getArgOperand(0);
        Placeholder->eraseFromParent();
      }
    }
  }

  // Since we don't handle phi nodes, we can safely erase other users in
  // reverse order of how they were encountered.
  if (!isArgument) {
    for (Instruction *OtherUser : llvm::reverse(Analysis.otherUsers))
      OtherUser->eraseFromParent();
  }

  return true;
}

// tsymalla: Create an equivalent function placeholder of ToClone, without 
// any instructions, but use the scalarized argument list instead.
Function *IPSROAPass::createDummyFunctionWithScalarizedArgs(Function *ToClone) {
  SmallVector<Type *, 1> ArgTys;
  std::vector<std::string> ArgNames;

  LLVM_DEBUG(dbgs() << "Creating new function placeholder for: "
                    << ToClone->getName() << "\n");

  for (size_t ArgIdx = 0; ArgIdx < ToClone->arg_size(); ++ArgIdx) {
    Argument *Arg = ToClone->getArg(ArgIdx);
    if (m_ArgToScalarArgsMapping.find(ArgIdx) !=
        m_ArgToScalarArgsMapping.end()) {
      ScalarizationPlan &Plan = getPlan(Arg);
      for (size_t ScalarIdx = 0;
           ScalarIdx < m_ArgToScalarArgsMapping[ArgIdx].size(); ++ScalarIdx) {
        ArgTys.push_back(Plan.Scalars[ScalarIdx].T);
        ArgNames.push_back(
            (Twine(Arg->getName()) + ".sroa." + std::to_string(ScalarIdx))
                .str());
      }
    } else {
      ArgTys.push_back(Arg->getType());
      ArgNames.push_back(Twine(Arg->getName()).str());
    }
  }

  FunctionType *NewFuncTy =
      FunctionType::get(ToClone->getReturnType(), ArgTys, false);
  Function *NewFunc = Function::Create(NewFuncTy, ToClone->getLinkage());
  NewFunc->setCallingConv(ToClone->getCallingConv());
  NewFunc->setSubprogram(ToClone->getSubprogram());
  NewFunc->setDLLStorageClass(ToClone->getDLLStorageClass());
  NewFunc->setAttributes(ToClone->getAttributes());

  // Clone the metadata
  SmallVector<std::pair<unsigned, MDNode *>, 8> AllMD;
  ToClone->getAllMetadata(AllMD);

  for (auto &[Offset, MD] : AllMD) {
    NewFunc->setMetadata(Offset, MD);
  }

  // Insert the new function after the old one.
  ToClone->getParent()->getFunctionList().insertAfter(ToClone->getIterator(),
                                                      NewFunc);

  // Rename the arguments
  for (size_t Idx = 0; Idx < NewFunc->arg_size(); ++Idx) {
    Argument *Arg = NewFunc->getArg(Idx);
    if (m_ArgToScalarArgsMapping.find(Idx) != m_ArgToScalarArgsMapping.end()) {
      for (size_t ArgNameIdx = Idx;
           ArgNameIdx < Idx + m_ArgToScalarArgsMapping[Idx].size();
           ++ArgNameIdx) {
        Arg->setName(ArgNames[ArgNameIdx]);
      }
    } else {
      Arg->setName(ArgNames[Idx]);
    }
  }

  LLVM_DEBUG(dbgs() << "Created new function placeholder: " << *NewFunc
                    << "\n");
  return NewFunc;
}
