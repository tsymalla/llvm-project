#ifndef LLVM_TRANSFORMS_IPO_IPSROA_H
#define LLVM_TRANSFORMS_IPO_IPSROA_H

#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class AllocaInst;
class Argument;
class DataLayout;
class DominatorTree;
class ReturnInst;
class Type;
class Value;
class LazyCallGraph;

class IPSROAPass : public llvm::PassInfoMixin<IPSROAPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &Module,
                              llvm::ModuleAnalysisManager &Analyses);

  struct Range {
    unsigned Offset;
    unsigned Size;

    void step() { Offset += Size; }
    unsigned peekStepVal() const { return Offset + Size; }
    void processSize(unsigned StepSize) {
      Offset += StepSize;
      Size -= StepSize;
    }

    bool operator<(const Range &RHS) const {
      if (Offset < RHS.Offset)
        return true;

      return false;
    }
  };

private:
  class BasicBlockOrderCache {
  private:
    llvm::DenseMap<llvm::Instruction *, unsigned> m_map;

  public:
    void reset() { m_map.clear(); }
    unsigned get(llvm::Instruction *inst);
  };

  /// Result of a preliminary, function-local analysis of uses of a given
  /// base pointer.
  struct ScalarizationAnalysis {
    struct Access {
      llvm::Instruction *Instruction;
      llvm::Argument *FuncCallArgument = nullptr;
      llvm::Type *T = nullptr;
      Range Range;
    };

    bool IsScalarizable = true;
    unsigned size =
        0; // size, in bytes, of the memory that is assumed to be present
    std::vector<Access> Accesses;
    std::vector<llvm::Instruction *> otherUsers;

    void addAccess(const Access &access);
  };

  struct Scalar {
    llvm::Type *T = nullptr;
    Range Range;

    bool input = false;
    bool output = false;
    bool outnull = false;

    bool operator<(const Scalar &RHS) const { return Range < RHS.Range; }
  };

  /// Plan for how to decompose a base pointer into scalars.
  struct ScalarizationPlan {
    std::vector<Scalar> Scalars;
  };

  llvm::Module *m_module = nullptr;
  const llvm::DataLayout *m_dataLayout;
  BasicBlockOrderCache m_BasicBlockOrder;
  llvm::LazyCallGraph *m_callGraph = nullptr;
  llvm::SmallVector<llvm::Function *, 3> m_functions;

  ScalarizationAnalysis &getAnalysis(llvm::Value *Value);
  ScalarizationPlan &getPlan(llvm::Value *Value);
  void collectFunctions();
  void scanAllocas();
  bool shouldHandleAlloca(llvm::AllocaInst *Alloca) const;
  void buildScalarizationPlans(llvm::ModuleAnalysisManager &Analyses);
  void findScalarizableArguments();
  bool runScalarization(llvm::ModuleAnalysisManager &Analyses);
  ScalarizationAnalysis analyze(llvm::Value *pointer);
  void preplan(llvm::Value *pointer);
  bool scalarize(
      llvm::DominatorTree &dt, llvm::Value *pointer,
      llvm::ArrayRef<llvm::Value *> inputs,
      llvm::DenseMap<llvm::ReturnInst *, std::vector<llvm::Value *>> *outputs,
      bool isArgument = false, llvm::Function *ClonedFunc = nullptr);

  llvm::Function *
  createDummyFunctionWithScalarizedArgs(llvm::Function *toClone);

  /// Analysis of a given base pointer.
  llvm::DenseMap<llvm::Value *, ScalarizationAnalysis> m_analysis;

  /// The plan for scalarizing a base pointer.
  llvm::DenseMap<llvm::Value *, ScalarizationPlan> m_plan;

  /// Candidate base pointers (allocas and arguments) by function.
  llvm::DenseMap<llvm::Function *, std::vector<llvm::Value *>>
      m_candidatePointers;

  /// Functions and their replaced, scalarized pendants.
  llvm::DenseMap<llvm::Function *, llvm::Function *> m_scalarizedFunctions;

  /// Maps the position of an argument in the non-scalarized function signature
  /// to the indices of the scalarized arguments for a single function.
  llvm::DenseMap<size_t, std::vector<size_t>> m_ArgToScalarArgsMapping;

  /// Maps the address of an argument in a non-scalarized function to its new
  /// position in the signature of the scalarized function.
  llvm::DenseMap<Argument *, size_t> m_OldArgToNewArgMapping;

  /// Used to replace the function calls mapping to an old function to a newly
  /// created function.
  llvm::DenseMap<llvm::Function *, std::vector<llvm::CallInst *>> m_FuncCalls;

  /// Maps an existing function to its clone (with scalarized arguments).
  llvm::DenseMap<llvm::Function *, llvm::Function *> m_OldToNewFuncMapping;
};

} // namespace llvm

#endif
