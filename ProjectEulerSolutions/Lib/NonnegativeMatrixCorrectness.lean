import ProjectEulerSolutions.Lib.NonnegativeMatrixCalls

/-! The source validator accepts exactly matrices whose integer entries are nonnegative. -/
namespace ProjectEulerSolutions.NonnegativeMatrixCalls
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false
variable {program : Module}

theorem values_then (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.list (encode values))) (position : Nat) (env : HeapCalls.Env)
    (following : List Stmt) (answer : Mutable.Value)
    (continued : ∀ after, BlockReturns program after following heap answer heap) :
    BlockReturns program env
      (.sequenceStepLoop "value" (.ref address) position valueBody [] :: following)
      heap (if ∀ value ∈ values.drop position, 0 ≤ value then answer else .scalar (.bool false)) heap := by
  have source : Mutable.sequence (.ref address) heap = some (.ok (encode values), heap) := by
    simp [Mutable.sequence, found]
  by_cases active : position < values.length
  · let value := values[position]
    let entered := setLocal env "value" (.scalar (.int value))
    have item : (encode values)[position]? = some (.scalar (.int value)) := by
      simp [encode, List.getElem?_eq_getElem active, value]
    have tested : Evaluates program entered (.compare (.name "value") [(.lt, .constant (.int 0))])
        heap (.scalar (.bool (decide (value < 0)))) heap := by
      apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    by_cases negative : value < 0
    · have rejected : ¬ ∀ v ∈ values.drop position, 0 ≤ v := by
        intro accepted
        have := accepted value (by rw [List.drop_eq_getElem_cons active]; exact List.mem_cons_self)
        omega
      rw [if_neg rejected]
      apply BlockReturns.sequence_step (source := .ref address) source item
      exact BlockReturns.branch (flag := true) (by simpa only [negative, decide_true] using tested) rfl
        (BlockReturns.return_value Evaluates.constant)
    · have accepted : 0 ≤ value := by omega
      have step : BlockContinues program entered valueBody heap entered heap :=
        BlockContinues.branch (flag := false) (by simpa only [negative, decide_false] using tested) rfl BlockContinues.nil
      have next := values_then heap address values found (position + 1) entered following answer continued
      apply BlockReturns.sequence_step (source := .ref address) source item
      simpa only [List.drop_eq_getElem_cons active, List.forall_mem_cons, ← show values[position] = value from rfl,
        show 0 ≤ values[position] from accepted, true_and, valueBody, Calls.loopContinuation,
        Calls.loopControl, Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true, if_false] using step.returns next
  · have gone : values.drop position = [] := List.drop_eq_nil_iff.mpr (by omega)
    simp only [gone, List.not_mem_nil, false_implies, implies_true, if_true]
    exact BlockReturns.sequence_done source
      (by simp [encode, List.getElem?_eq_none (by omega : values.length ≤ position)]) (continued env)
termination_by values.length - position
decreasing_by omega

theorem rows_result (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (position : Nat) (env : HeapCalls.Env) :
    BlockReturns program env
      [.sequenceStepLoop "row" (.ref address) position rowBody [], .return (.constant (.bool true))]
      heap (.scalar (.bool (decide (∀ pair ∈ rows.drop position, ∀ value ∈ pair.2, 0 ≤ value)))) heap := by
  have source : Mutable.sequence (.ref address) heap =
      some (.ok (rows.map (fun pair => .ref pair.1)), heap) := by
    simp [Mutable.sequence, found]
  by_cases active : position < rows.length
  · let pair := rows[position]
    let entered := setLocal env "row" (.ref pair.1)
    have item : (rows.map (fun pair => Mutable.Value.ref pair.1))[position]? = some (.ref pair.1) := by
      simp [List.getElem?_eq_getElem active, pair]
    have rowSource : Mutable.sequence (.ref pair.1) heap = some (.ok (encode pair.2), heap) := by
      simp [Mutable.sequence, reads pair (List.getElem_mem active)]
    have next := fun after => rows_result heap address rows found reads (position + 1) after
    have scanned := values_then (program := program) heap pair.1 pair.2 (reads pair (List.getElem_mem active)) 0 entered
      [.sequenceStepLoop "row" (.ref address) (position + 1) rowBody [], .return (.constant (.bool true))]
      _ next
    apply BlockReturns.sequence_step (source := .ref address) source item
    apply BlockReturns.for_sequence (by intros expressions impossible; cases impossible)
      (Evaluates.lookup rfl) rowSource
    rw [List.drop_eq_getElem_cons active]
    by_cases accepted : ∀ value ∈ rows[position].2, 0 ≤ value
    · rw [if_pos (show ∀ value ∈ pair.2.drop 0, 0 ≤ value by simpa only [List.drop_zero] using accepted)] at scanned
      simpa only [List.forall_mem_cons, Bool.decide_and, decide_eq_true accepted, Bool.true_and] using scanned
    · simpa only [pair, List.drop_zero, List.forall_mem_cons, accepted, false_and, if_false, decide_false] using scanned
  · have gone : rows.drop position = [] := List.drop_eq_nil_iff.mpr (by omega)
    simp only [gone, List.not_mem_nil, false_implies, implies_true, decide_true]
    exact BlockReturns.sequence_done source
      (by simp [List.getElem?_eq_none (by omega : rows.length ≤ position)])
      (BlockReturns.return_value Evaluates.constant)
termination_by rows.length - position
decreasing_by omega

theorem check_correct
    (source : findFunctionIn program.body "has_nonnegative_entries" = some (arguments, checkCode))
    (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) :
    ReturnsValue program "has_nonnegative_entries" [.ref address] heap
      (.scalar (.bool (decide (∀ pair ∈ rows, ∀ value ∈ pair.2, 0 ≤ value)))) heap := by
  apply BlockReturns.function source rfl
  apply BlockReturns.for_sequence (items := rows.map (fun pair => .ref pair.1))
    (by intros expressions impossible; cases impossible) (Evaluates.lookup rfl)
    (by simp [Mutable.sequence, found])
  exact rows_result heap address rows found reads 0 _

theorem domain_of_condition
    (source : findFunctionIn program.body "has_nonnegative_entries" = some (arguments, checkCode))
    (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (requires : HeapCalls.Condition program (matrixEnv address) checkExpression heap) :
    ∀ pair ∈ rows, ∀ value ∈ pair.2, 0 ≤ value := by
  have evaluated : Evaluates program (matrixEnv address) checkExpression heap
      (.scalar (.bool (decide (∀ pair ∈ rows, ∀ value ∈ pair.2, 0 ≤ value)))) heap :=
    Evaluates.call (.cons (Evaluates.lookup rfl) .nil) (check_correct source heap address rows found reads)
  have accepted := evaluated.condition_iff.mp requires
  simpa using accepted

end ProjectEulerSolutions.NonnegativeMatrixCalls
