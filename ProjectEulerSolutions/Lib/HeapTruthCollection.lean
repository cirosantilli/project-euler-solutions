import lang.py.FormalSW.TruthCalls
import ProjectEulerSolutions.Lib.HeapCollection

/-! Lazy universal predicates may allocate fresh objects while preserving their source sequence. -/
namespace ProjectEulerSolutions.HeapTruthCollection
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

variable {globals : Mutable.Globals}

private theorem loop {α : Type} (program : Module) (values : List α) (encode : α → Mutable.Value)
    (predicate : α → Bool) (name : String) (element : Expr) (base : Mutable.Heap) (address : Nat)
    (found : base[address]?=some (.list (values.map encode)))
    (invariant : HeapCalls.Env → Prop)
    (preserved : ∀ inner item, invariant inner → invariant (setLocal inner name (encode item)))
    (callback : ∀ garbage inner item, invariant inner → inner name=some (encode item) →
      ∃ extra, Evaluates (globals := globals) program inner element (base++garbage)
        (.scalar (.bool (predicate item))) (base++garbage++extra))
    (remaining : List α) (position : Nat) (suffix : values.drop position=remaining)
    (garbage : Mutable.Heap) (env : HeapCalls.Env) (valid : invariant env) :
    ∃ extra, BlockReturns (globals := globals) program env
      [.sequenceStepLoop name (.ref address) position (truthBody true element) [],
        .return (.constant (.bool true))] (base++garbage)
      (.scalar (.bool (remaining.all predicate))) (base++garbage++extra) := by
  induction remaining generalizing position env garbage with
  | nil =>
      refine ⟨[], ?_⟩
      simp only [List.all_nil, List.append_nil]
      apply BlockReturns.sequence_done (items := values.map encode)
        (by simp [sequenceValue, Mutable.sequence, HeapCollection.prefix_found base garbage address _ found])
      · have missing : values[position]?=none := by
          simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
        simp [missing]
      · exact BlockReturns.return_value Evaluates.constant
  | cons head tail ih =>
      let bound := setLocal env name (encode head)
      have validBound : invariant bound := preserved env head valid
      obtain ⟨allocated, tested⟩ := callback garbage bound head validBound (by simp [bound, setLocal])
      have item : values[position]?=some head := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      cases flag : predicate head with
      | false =>
          refine ⟨allocated, ?_⟩
          simp only [List.all_cons, flag, Bool.false_and]
          apply BlockReturns.sequence_step (items := values.map encode) (value := encode head)
            (by simp [sequenceValue, Mutable.sequence, HeapCollection.prefix_found base garbage address _ found])
            (by simp [item])
          apply BlockReturns.branch (by simpa only [flag] using tested) rfl
          exact BlockReturns.return_value Evaluates.constant
      | true =>
          obtain ⟨extra, rest⟩ := ih (position+1) (by simp [List.drop_add_one_eq_tail_drop, suffix])
            (garbage++allocated) bound validBound
          refine ⟨allocated++extra, ?_⟩
          simp only [List.all_cons, flag, Bool.true_and]
          apply BlockReturns.sequence_step (items := values.map encode) (value := encode head)
            (by simp [sequenceValue, Mutable.sequence, HeapCollection.prefix_found base garbage address _ found])
            (by simp [item])
          apply BlockReturns.branch (by simpa only [flag] using tested) rfl
          simpa only [List.append_assoc, truthBody, Calls.loopContinuation, Calls.loopControl,
            Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true, if_false, List.nil_append] using rest

/-- Captured locals remain available after each fresh predicate call, including on early failure. -/
theorem all_map {α : Type} (program : Module) (values : List α) (encode : α → Mutable.Value)
    (predicate : α → Bool) (name : String) (element iterable : Expr)
    (heap base : Mutable.Heap) (address : Nat) (env : HeapCalls.Env)
    (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (notEnumerate : ∀ expressions, iterable ≠ .call (.name "enumerate") expressions)
    (notZip : ∀ expressions, iterable ≠ .zipIterable expressions)
    (evaluated : Evaluates (globals := globals) program env iterable heap (.ref address) base)
    (found : base[address]?=some (.list (values.map encode)))
    (invariant : HeapCalls.Env → Prop)
    (preserved : ∀ inner item, invariant inner → invariant (setLocal inner name (encode item)))
    (callback : ∀ garbage inner item, invariant inner → inner name=some (encode item) →
      ∃ extra, Evaluates (globals := globals) program inner element (base++garbage)
        (.scalar (.bool (predicate item))) (base++garbage++extra))
    (valid : invariant (comprehensionTargetsEnv env [(.name name, iterable, [])])) :
    ∃ extra, Evaluates (globals := globals) program env
      (.consumeGenerator (.builtinCall .all []) element [(.name name, iterable, [])]) heap
      (.scalar (.bool (values.all predicate))) (base++extra) := by
  obtain ⟨extra, run⟩ := loop program values encode predicate name element base address found invariant
    preserved callback values 0 rfl [] _ valid
  refine ⟨extra, ?_⟩
  apply Evaluates.truth_generator_sequence (universal := true) (items := values.map encode) ordinary notEnumerate notZip evaluated
    (by simp [Mutable.sequence, found])
  simpa only [List.append_nil] using run

/-- Pure Boolean prefixes retain Python's short circuit before an allocating final operand. -/
theorem and_prefix (program : Module) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (conditions : List (Expr×Bool)) (last : Expr) (answer : Bool)
    (tests : ∀ expression value, (expression,value) ∈ conditions →
      Evaluates (globals := globals) program env expression heap (.scalar (.bool value)) heap)
    (final : ∃ extra, Evaluates (globals := globals) program env last heap (.scalar (.bool answer)) (heap++extra)) :
    ∃ extra, Evaluates (globals := globals) program env
      (.boolOp .and (conditions.map Prod.fst++[last])) heap
      (.scalar (.bool (conditions.all Prod.snd && answer))) (heap++extra) := by
  induction conditions with
  | nil =>
      obtain ⟨extra, evaluated⟩ := final
      exact ⟨extra, Evaluates.boolean_last evaluated rfl⟩
  | cons condition conditions ih =>
      rcases condition with ⟨expression, value⟩
      have first := tests expression value (by simp)
      cases value with
      | false =>
          refine ⟨[], ?_⟩
          simp only [List.map_cons, List.cons_append, List.all_cons, Bool.false_and, List.append_nil]
          exact Evaluates.boolean_stop first rfl rfl
      | true =>
          obtain ⟨extra, rest⟩ := ih (by intro expression value member; exact tests expression value (by simp [member]))
          refine ⟨extra, ?_⟩
          simp only [List.map_cons, List.cons_append, List.all_cons, Bool.true_and]
          cases conditions <;> exact Evaluates.boolean_next first rfl rfl rest

end ProjectEulerSolutions.HeapTruthCollection
