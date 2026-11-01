import ProjectEulerSolutions.Lib.RectangularMatrixCalls

/-! Total, heap-preserving validation of nonnegative nested integer lists. -/
namespace ProjectEulerSolutions.NonnegativeMatrixCalls
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RectangularMatrixCalls
set_option Elab.async false
variable {program : Module}

abbrev valueBody : List Stmt :=
  [.ifThen (.compare (.name "value") [(.lt, .constant (.int 0))]) [.return (.constant (.bool false))] []]
abbrev rowBody : List Stmt := [.forIn (.name "value") (.name "row") valueBody []]
abbrev checkCode : List Stmt :=
  [.forIn (.name "row") (.name "matrix") rowBody [], .return (.constant (.bool true))]
abbrev checkExpression : Expr := .call (.name "has_nonnegative_entries") [.name "matrix"]

theorem values_check (heap : Mutable.Heap) (address : Nat) (values : List Int)
    (found : heap[address]? = some (.list (encode values))) (position : Nat) (env : HeapCalls.Env) :
    (∃ finalEnv, BlockContinues program env [.sequenceStepLoop "value" (.ref address) position valueBody []]
      heap finalEnv heap) ∨
    (∀ following, BlockReturns program env (.sequenceStepLoop "value" (.ref address) position valueBody [] :: following)
      heap (.scalar (.bool false)) heap) := by
  have source : Mutable.sequence (sequenceValue (.ref address)) heap = some (.ok (encode values), heap) := by
    simp [sequenceValue, Mutable.sequence, found]
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
    · right
      intro following
      apply BlockReturns.sequence_step source item
      exact BlockReturns.branch (flag := true) (by simpa only [negative, decide_true] using tested) rfl
        (BlockReturns.return_value Evaluates.constant)
    · have step : BlockContinues program entered valueBody heap entered heap :=
        BlockContinues.branch (flag := false) (by simpa only [negative, decide_false] using tested) rfl BlockContinues.nil
      rcases values_check heap address values found (position + 1) entered with ⟨finalEnv, continued⟩ | rejected
      · left
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.sequence_step source item
        simpa [valueBody, Calls.loopContinuation, Calls.loopControl] using step.append continued
      · right
        intro following
        apply BlockReturns.sequence_step source item
        simpa [valueBody, Calls.loopContinuation, Calls.loopControl] using step.returns (rejected following)
  · left
    refine ⟨env, ?_⟩
    exact BlockContinues.sequence_done source
      (by simp [encode, List.getElem?_eq_none (by omega : values.length ≤ position)]) BlockContinues.nil
termination_by values.length - position
decreasing_by omega

theorem rows_check (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (position : Nat) (env : HeapCalls.Env) :
    ∃ flag, BlockReturns program env
      [.sequenceStepLoop "row" (.ref address) position rowBody [], .return (.constant (.bool true))]
      heap (.scalar (.bool flag)) heap := by
  have source : Mutable.sequence (sequenceValue (.ref address)) heap =
      some (.ok (rows.map (fun pair => .ref pair.1)), heap) := by
    simp [sequenceValue, Mutable.sequence, found]
  by_cases active : position < rows.length
  · let pair := rows[position]
    let entered := setLocal env "row" (.ref pair.1)
    have item : (rows.map (fun pair => Mutable.Value.ref pair.1))[position]? = some (.ref pair.1) := by
      simp [List.getElem?_eq_getElem active, pair]
    have rowSource : Mutable.sequence (.ref pair.1) heap = some (.ok (encode pair.2), heap) := by
      simp [Mutable.sequence, reads pair (List.getElem_mem active)]
    rcases values_check (program := program) heap pair.1 pair.2 (reads pair (List.getElem_mem active)) 0 entered with
      ⟨after, checked⟩ | rejected
    · have checkedRow : BlockContinues program entered rowBody heap after heap := by
        exact BlockContinues.for_sequence (by intros expressions impossible; cases impossible)
          (Evaluates.lookup rfl) rowSource checked
      obtain ⟨flag, continued⟩ := rows_check heap address rows found reads (position + 1) after
      refine ⟨flag, ?_⟩
      apply BlockReturns.sequence_step source item
      simpa [rowBody, Calls.loopContinuation, Calls.loopControl] using checkedRow.returns continued
    · refine ⟨false, ?_⟩
      apply BlockReturns.sequence_step source item
      apply BlockReturns.for_sequence (by intros expressions impossible; cases impossible)
        (Evaluates.lookup rfl) rowSource
      exact rejected _
  · refine ⟨true, ?_⟩
    exact BlockReturns.sequence_done source
      (by simp [List.getElem?_eq_none (by omega : rows.length ≤ position)])
      (BlockReturns.return_value Evaluates.constant)
termination_by rows.length - position
decreasing_by omega

theorem check_result
    (source : findFunctionIn program.body "has_nonnegative_entries" = some (arguments, checkCode))
    (heap : Mutable.Heap) (address : Nat) (rows : MatrixRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) :
    ∃ flag, ReturnsValue program "has_nonnegative_entries" [.ref address] heap (.scalar (.bool flag)) heap := by
  obtain ⟨flag, loop⟩ := rows_check heap address rows found reads 0 (argumentEnv arguments [.ref address])
  refine ⟨flag, ?_⟩
  apply BlockReturns.function source rfl
  exact BlockReturns.for_sequence (items := rows.map (fun pair => .ref pair.1))
    (by intros expressions impossible; cases impossible) (Evaluates.lookup rfl)
    (by simp [Mutable.sequence, found]) loop

theorem condition_supported
    (source : findFunctionIn program.body "has_nonnegative_entries" = some (arguments, checkCode))
    (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    (HeapCalls.evaluate program (matrixEnv address) checkExpression heap).Supported := by
  obtain ⟨rows, found, reads⟩ := decode_rows heap address typed
  obtain ⟨flag, returned⟩ := check_result source heap address rows found reads
  have evaluated : Evaluates program (matrixEnv address) checkExpression heap (.scalar (.bool flag)) heap :=
    Evaluates.call (.cons (Evaluates.lookup rfl) .nil) returned
  exact Execution.supported_of_finishes evaluated.finishes

end ProjectEulerSolutions.NonnegativeMatrixCalls
