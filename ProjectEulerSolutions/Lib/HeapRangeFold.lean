import lang.py.FormalSW.HeapCalls

namespace ProjectEulerSolutions.HeapRangeFold
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
set_option Elab.async false
universe u
variable {globals : Mutable.Globals}

def Frame (changed : List String) (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ∉ changed → after name = before name

theorem Frame.refl (changed : List String) (env : HeapCalls.Env) : Frame changed env env :=
  fun _ _ => rfl

theorem Frame.trans {changed : List String} {first second third : HeapCalls.Env}
    (left : Frame changed first second) (right : Frame changed second third) : Frame changed first third :=
  fun name outside => (right name outside).trans (left name outside)

theorem Frame.updated (changed : List String) (env : HeapCalls.Env) (name : String)
    (value : Mutable.Value) (written : name ∈ changed) : Frame changed env (setLocal env name value) := by
  intro other outside
  have different : other ≠ name := by intro same; subst other; exact outside written
  simp [setLocal, different]

/-- A structural range fold permits arbitrary finite allocation by each source iteration.
The callback proves the actual body, the state invariant, preservation of the caller heap,
and every local outside the declared write set. -/
theorem loop {State : Type u} (program : Module) (name : String) (body : List Stmt)
    (noControl : Calls.loopControl body = false) (changed : List String) (written : name ∈ changed)
    (update : State → Nat → State) (Invariant : Mutable.Heap → HeapCalls.Env → State → Prop)
    (lower upper : Nat)
    (bindInvariant : ∀ heap env state value, Invariant heap env state →
      lower ≤ value → value < upper →
      Invariant heap (setLocal env name (.scalar (.int value))) state)
    (cell : ∀ heap env state value, Invariant heap env state → lower ≤ value → value < upper →
      env name = some (.scalar (.int value)) →
      ∃ afterEnv finalHeap, BlockContinues program env body heap afterEnv finalHeap globals ∧
        Invariant finalHeap afterEnv (update state value) ∧
        (∃ extra, finalHeap = heap ++ extra) ∧ Frame changed env afterEnv)
    (count first : Nat) (start : lower ≤ first) (stop : first + count = upper)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (state : State) (safe : Invariant heap env state) :
    ∃ afterEnv finalHeap,
      BlockContinues program env [.rangeStepLoop name first upper 1 body []]
        heap afterEnv finalHeap globals ∧
      Invariant finalHeap afterEnv ((List.range' first count).foldl update state) ∧
      (∃ extra, finalHeap = heap ++ extra) ∧ Frame changed env afterEnv := by
  induction count generalizing first heap env state with
  | zero =>
      refine ⟨env, heap, ?_, safe, ⟨[], by simp⟩, Frame.refl changed env⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let bound := setLocal env name (.scalar (.int first))
      have boundSafe : Invariant heap bound state := bindInvariant heap env state first safe start (by omega)
      obtain ⟨middleEnv, middleHeap, executed, middleSafe, ⟨firstExtra, firstHeap⟩, firstFrame⟩ :=
        cell heap bound state first boundSafe start (by omega) (by simp [bound, setLocal])
      obtain ⟨afterEnv, finalHeap, continued, finalSafe, ⟨restExtra, restHeap⟩, restFrame⟩ :=
        ih (first + 1) (by omega) (by omega) middleHeap middleEnv (update state first) middleSafe
      refine ⟨afterEnv, finalHeap, ?_, ?_, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, noControl, Bool.false_eq_true, if_false,
          Int.natCast_add, Int.natCast_one] using executed.append continued
      · simpa only [List.range'_succ, List.foldl_cons] using finalSafe
      · exact ⟨firstExtra ++ restExtra, by rw [restHeap, firstHeap, List.append_assoc]⟩
      · exact (Frame.updated changed env name _ written).trans (firstFrame.trans restFrame)

end ProjectEulerSolutions.HeapRangeFold
