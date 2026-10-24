import solvers.«68.setup.source»
import ProjectEulerStatements.Lib.MagicRingReferenceScan
import ProjectEulerSolutions.Equiv.P68

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRing MagicRingTextCalls
set_option Elab.async false

theorem positive_result (n digits : Nat) (size : 3 ≤ n) (width : 3 * n ≤ digits)
    (heap : Mutable.Heap) (values : List Mutable.Value)
    (arguments : completeArguments solverArgs values = some [number n, number digits]) :
    ∃ finalHeap, ReturnsValue program "solve" values heap
      (string (referenceAnswer n digits)) finalHeap := by
  let env := HeapCalls.argumentEnv solverArgs [number n, number digits]
  obtain ⟨garbage, pooled⟩ := labels_result n heap env rfl
  let labelsHeap := heap ++ (.list (encode (labels n)) :: garbage)
  let labelsEnv := setLocal env "labels" (.ref heap.length)
  let numsHeap := labelsHeap ++ [.set (IntegerSets.entries (ints (labels n)))]
  let numsEnv := setLocal labelsEnv "nums" (.ref labelsHeap.length)
  let widthEnv := setLocal numsEnv "total_digits" (number ((labels n).map digitWidth).sum)
  let totalEnv := setLocal widthEnv "total" (number (n * (2 * n + 1)))
  let readyEnv := setLocal totalEnv "best" (string "")
  have ready : OuterReady n digits "" heap.length labelsHeap.length readyEnv := by
    constructor <;> rfl
  have labelsFound : labelsHeap[heap.length]? = some (.list (encode (labels n))) := by
    simp [labelsHeap]
  have numsCreated : Evaluates program labelsEnv (.builtinCall .set [.name "labels"])
      labelsHeap (.ref labelsHeap.length) numsHeap :=
    Evaluates.set_integer_sequence (tuple := false) (items := ints (labels n))
      (Evaluates.lookup rfl)
      (by simpa only [Mutable.scalarSequence, IntegerSets.encoded, ints, encode,
        List.map_map, Function.comp_def, Bool.false_eq_true, if_false] using labelsFound)
  obtain ⟨finalHeap, scanned⟩ := selection_loop_result n digits (by omega) numsHeap readyEnv
    heap.length labelsHeap.length
    (ProjectEulerSolutions.HeapCollection.prefix_found labelsHeap _ _ _ labelsFound)
    (by simp [numsHeap]) ready
  rw [answers_equal n digits (by omega)] at scanned
  have tested := guard_result n digits heap env rfl rfl
  have guarded : ¬ ((n : Int) < 3 ∨ (digits : Int) < 3 * n) := by omega
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function solver_function_source arguments
  change BlockReturns program env solverCode heap (string (referenceAnswer n digits)) finalHeap
  rw [solver_code_source]
  apply BlockReturns.branch (flag := false)
    (by simpa only [guarded, decide_false] using tested) rfl
  apply BlockReturns.assign pooled
  apply BlockReturns.assign numsCreated
  apply BlockReturns.assign (total_width_result n numsHeap numsEnv rfl)
  apply BlockReturns.assign (label_total_result n numsHeap widthEnv rfl)
  apply BlockReturns.assign Evaluates.constant
  exact scanned

theorem solve.result (n digits : Int) (heap : Mutable.Heap) (values : List Mutable.Value)
    (arguments : completeArguments solverArgs values = some [integer n, integer digits]) :
    ∃ finalHeap, ReturnsValue program "solve" values heap
      (string (referenceResult n digits)) finalHeap := by
  by_cases rejected : n < 3 ∨ digits < 3 * n
  · refine ⟨heap, ?_⟩
    have tested := guard_result n digits heap
      (HeapCalls.argumentEnv solverArgs [integer n, integer digits]) rfl rfl
    apply BlockReturns.function solver_function_source arguments
    change BlockReturns program (HeapCalls.argumentEnv solverArgs [integer n, integer digits])
      solverCode heap (string (referenceResult n digits)) heap
    rw [solver_code_source]
    apply BlockReturns.branch (flag := true)
      (by simpa only [rejected, decide_true] using tested) rfl
    simpa only [referenceResult, if_pos rejected] using
      (BlockReturns.return_value (module := program) (value := string "") Evaluates.constant)
  · have nCast : (n.toNat : Int) = n := Int.toNat_of_nonneg (by omega)
    have digitsCast : (digits.toNat : Int) = digits := Int.toNat_of_nonneg (by omega)
    have size : 3 ≤ n.toNat := by omega
    have width : 3 * n.toNat ≤ digits.toNat := by omega
    obtain ⟨finalHeap, executed⟩ := positive_result n.toNat digits.toNat size width heap values
      (by simpa only [number, nCast, digitsCast, integer] using arguments)
    exact ⟨finalHeap, by simpa only [referenceResult, if_neg rejected, number,
      nCast, digitsCast, integer] using executed⟩

theorem solve.full_result (n digits : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [integer n, integer digits] heap
      (string (referenceResult n digits)) finalHeap := solve.result n digits heap _ rfl

theorem solve.model_result (n digits : Int) (heap : Mutable.Heap) (values : List Mutable.Value)
    (arguments : completeArguments solverArgs values = some [integer n, integer digits]) :
    ∃ finalHeap, ReturnsValue program "solve" values heap
      (string (ProjectEulerSolutions.P68.solve n digits)) finalHeap := by
  rw [← ProjectEulerSolutions.P68.equiv_at n digits]
  exact solve.result n digits heap values arguments

theorem solve.one_argument_result (n : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [integer n] heap
      (string (referenceResult n 16)) finalHeap := solve.result n 16 heap _ rfl

theorem solve.default_result (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [] heap
      (string (referenceResult 5 16)) finalHeap := solve.result 5 16 heap _ rfl

end solvers.«68»
