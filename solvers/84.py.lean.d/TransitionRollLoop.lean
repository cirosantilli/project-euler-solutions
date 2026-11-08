import solvers.«84.py.lean.d».TransitionRollState

namespace solvers.«84».transition_row.rolls
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

theorem scan (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (source tableRoot root pos doubles position : Nat) (original : List outcomes.Outcome) (entries : Entries)
    (sourceView : outcomes.View heap source original) (table : Table heap tableRoot) (found : Found heap root entries)
    (getTable : env "landings" = some (.ref tableRoot)) (getRow : env "row" = some (.ref root))
    (getPos : env "pos" = some (number pos)) (getDoubles : env "doubles" = some (number doubles))
    (nonnegative : ∀ outcome ∈ original, 0 ≤ outcome.move ∧ 0 ≤ outcome.count) :
    ∃ (result : Entries) (finalHeap : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env [cursor source position] heap finalEnv finalHeap globals ∧ Found finalHeap root result ∧
      items.HeapFrame heap finalHeap root ∧ LocalsFrame env finalEnv ∧
      ∀ target, ((lookup result target).getD 0 : Int) =
        ((lookup entries target).getD 0 : Int) + total pos doubles (original.drop position) target := by
  by_cases active : position < original.length
  · let outcome := original[position]
    have member : outcome ∈ original := List.getElem_mem active
    obtain ⟨values, address, present, _, selected, tupleFound⟩ := sourceView.at position active
    obtain ⟨updated, middle, middleEnv, ran, middleFound, frame, localFrame, nextCounts⟩ :=
      step_result env heap globals tableRoot root pos doubles outcome entries table found getTable getRow getPos getDoubles
        (nonnegative outcome member).1 (nonnegative outcome member).2
    obtain ⟨result, finalHeap, finalEnv, continued, finalFound, finalFrame, finalLocals, finalCounts⟩ :=
      scan middleEnv middle globals source tableRoot root pos doubles (position + 1) original updated
        (sourceView.frame entries found frame) (table.frame entries found frame) middleFound
        ((localFrame _ (by decide)).trans getTable) ((localFrame _ (by decide)).trans getRow)
        ((localFrame _ (by decide)).trans getPos) ((localFrame _ (by decide)).trans getDoubles) nonnegative
    refine ⟨result, finalHeap, finalEnv, ?_, finalFound, ?_, ?_, ?_⟩
    · apply BlockContinues.unpack_step (items := values) (value := .ref address) (values := outcome.items)
        (by simp [sequenceValue, Mutable.sequence, present]) selected (by simp [Mutable.sequence, tupleFound, outcome]) rfl
      simpa only [show Calls.loopContinuation code (cursor source (position + 1)) = cursor source (position + 1) from rfl]
        using ran.append continued
    · intro address object excluded initial
      exact finalFrame address object excluded (frame address object excluded initial)
    · intro name keep
      exact (finalLocals name keep).trans (localFrame name keep)
    · intro target
      rw [finalCounts, nextCounts]
      rw [List.drop_eq_getElem_cons active]
      simp only [total, List.map_cons, List.sum_cons, outcome, Int.add_assoc]
  · obtain ⟨values, present, represented⟩ := sourceView
    have exhausted : original.length ≤ position := by omega
    have noValue : values[position]? = none := List.getElem?_eq_none (by rw [← represented.length_eq]; exact exhausted)
    refine ⟨entries, heap, env, ?_, found, ?_, ?_, ?_⟩
    · exact BlockContinues.unpack_done (items := values) (by simp [sequenceValue, Mutable.sequence, present]) noValue BlockContinues.nil
    · intro address object _ present
      exact present
    · intro name _
      rfl
    · intro target
      simp [total, List.drop_eq_nil_of_le exhausted]
termination_by original.length - position
decreasing_by omega

/-- The complete source outcomes loop realizes the sum of all weighted roll contributions. -/
theorem result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (source tableRoot root pos doubles : Nat) (original : List outcomes.Outcome) (entries : Entries)
    (sourceView : outcomes.View heap source original) (table : Table heap tableRoot) (found : Found heap root entries)
    (getSource : env "outcomes" = some (.ref source)) (getTable : env "landings" = some (.ref tableRoot))
    (getRow : env "row" = some (.ref root)) (getPos : env "pos" = some (number pos))
    (getDoubles : env "doubles" = some (number doubles))
    (nonnegative : ∀ outcome ∈ original, 0 ≤ outcome.move ∧ 0 ≤ outcome.count) :
    ∃ (result : Entries) (finalHeap : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env [loop] heap finalEnv finalHeap globals ∧ Found finalHeap root result ∧
      items.HeapFrame heap finalHeap root ∧ LocalsFrame env finalEnv ∧
      ∀ target, ((lookup result target).getD 0 : Int) = ((lookup entries target).getD 0 : Int) + total pos doubles original target := by
  obtain ⟨result, finalHeap, finalEnv, ran, finalFound, frame, localFrame, counts⟩ :=
    scan env heap globals source tableRoot root pos doubles 0 original entries sourceView table found
      getTable getRow getPos getDoubles nonnegative
  obtain ⟨values, present, _⟩ := sourceView
  refine ⟨result, finalHeap, finalEnv, ?_, finalFound, frame, localFrame, by simpa only [List.drop_zero] using counts⟩
  exact BlockContinues.for_unpack (items := values) (by intro expressions; simp) rfl (Evaluates.lookup getSource)
    (by simp [Mutable.sequence, present]) ran

end solvers.«84».transition_row.rolls
