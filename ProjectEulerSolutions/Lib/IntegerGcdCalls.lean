import ProjectEulerSolutions.Lib.HeapSequenceFold
import ProjectEulerSolutions.Lib.IntegerGcdRow

/-! Source execution of signed integer gcd folds over live lists. -/
namespace ProjectEulerSolutions.IntegerGcdCalls
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def body (name : String) : List Stmt :=
  [.assign [.name "divisor"] (.call (.name "gcd") [.name "divisor", .name name])]

/-- Every iteration uses Python's nonnegative gcd, including negative and zero entries. -/
theorem result (module : Module) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (absent : findFunctionIn module.body "gcd" = none)
    (libraryFound : libraryFunction module "gcd" = some Math.gcdCall)
    (name : String) (different : name ≠ "divisor") (address : Nat) (values : List Int) (initial : Int)
    (getDivisor : env "divisor" = some (.scalar (.int initial)))
    (iterable : Expr) (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (evaluated : Evaluates module env iterable heap (.ref address) heap globals)
    (found : heap[address]? = some (.list (values.map fun value => .scalar (.int value)))) :
    ∃ afterEnv, BlockContinues module env [.forIn (.name name) iterable (body name) []] heap afterEnv heap globals ∧
      afterEnv "divisor" = some (.scalar (.int (values.foldl (fun divisor value => (Int.gcd divisor value : Int)) initial))) ∧
      HeapRangeFold.Frame [name, "divisor"] env afterEnv := by
  let invariant := fun current (locals : HeapCalls.Env) (divisor : Int) =>
    current = heap ∧ locals "divisor" = some (.scalar (.int divisor))
  obtain ⟨afterEnv, finalHeap, executed, safe, _, frame⟩ := HeapSequenceFold.for_list
    module name (body name) rfl [name, "divisor"] (by simp)
    (fun value : Int => Mutable.Value.scalar (.int value)) (fun divisor value => (Int.gcd divisor value : Int))
    invariant address values
    (by
      intro current locals divisor value safe _
      exact ⟨safe.1, by simpa [setLocal, Ne.symm different] using safe.2⟩)
    (by
      intro current locals divisor value safe _ getValue
      refine ⟨setLocal locals "divisor" (.scalar (.int (Int.gcd divisor value))), current, ?_, ?_,
        ⟨[], by simp⟩, HeapRangeFold.Frame.updated _ _ _ _ (by simp)⟩
      · apply BlockContinues.assign (Evaluates.library_call (.cons (.lookup safe.2) (.cons (.lookup getValue) .nil)) absent ?_)
        simp [HeapCalls.library, libraryFound, HeapCalls.scalarResult, Math.gcdCall_pair, Int.gcd_def]
      · exact ⟨safe.1, rfl⟩)
    heap env initial ⟨rfl, getDivisor⟩ iterable ordinary evaluated found
  obtain ⟨same, getFinal⟩ := safe
  subst finalHeap
  exact ⟨afterEnv, executed, getFinal, frame⟩

end ProjectEulerSolutions.IntegerGcdCalls
