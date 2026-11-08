import ProjectEulerSolutions.Lib.HeapMemoBlock
import Mathlib.Data.List.Range

/-! Structural range folds whose callbacks allocate or reuse cached tuples. -/
namespace ProjectEulerSolutions.HeapMemoBlock
open FormalSW.PyAst HeapCalls HeapMemoCalls
set_option Elab.async false
universe u
variable {module : Module} {spec : Specification} {globals : Mutable.Globals}

private def rangeArgumentsState (name : String) (env : HeapCalls.Env) (done : List Mutable.Value)
    (expressions : List Expr) (body otherwise following : List Stmt)
    (stack : List HeapCalls.Frame) (heap : Mutable.Heap) : HeapCalls.State :=
  match expressions with
  | [] => ⟨.startRange name done body otherwise following env, stack, heap⟩
  | first :: rest => ⟨.expr first env, .rangeArguments name done rest body otherwise following env :: stack, heap⟩

private theorem range_arguments
    (arguments : HeapCalls.Evaluates.Arguments module env expressions heap values after globals)
    (done : List Mutable.Value) (anchors : Nat) (stack : List HeapCalls.Frame) :
    HeapCalls.Reaches module anchors (rangeArgumentsState name env done expressions body otherwise following stack heap)
      ⟨.startRange name (done ++ values) body otherwise following env, stack, after⟩ globals := by
  induction arguments generalizing done with
  | nil => simp [rangeArgumentsState]; exact .refl
  | @cons expression heap value middle expressions values after globals head tail ih =>
      apply (head _ _).trans
      refine .next rfl ?_
      simpa [List.append_assoc] using ih (done ++ [value])

namespace Continues

theorem for_range (disjoint : CacheDisjoint module module)
    (arguments : HeapCalls.Evaluates.Arguments module env expressions heap values middle globals)
    (bounds : rangeBounds values = .ok (start, stop, stride)) (preserved : Cache.Preserves spec heap middle)
    (run : Continues module spec env [.rangeStepLoop name start stop stride body otherwise] middle post globals) :
    Continues module spec env [.forIn (.name name) (.call (.name "range") expressions) body otherwise] heap post globals := by
  intro following anchors stack
  have began : HeapCalls.Reaches module anchors
      ⟨.block (.forIn (.name name) (.call (.name "range") expressions) body otherwise :: following) env, stack, heap⟩
      ⟨.block (.rangeStepLoop name start stop stride body otherwise :: following) env, stack, middle⟩ globals :=
    .next rfl ((range_arguments arguments [] anchors stack).trans (.next (by simp [HeapCalls.step, bounds]) .refl))
  exact (Runs.from_heap (Extends.refl module) disjoint began preserved).bind
    (fun final same => by subst final; exact run following anchors stack)

theorem range_step (active : Calls.rangeContinues next stop stride = true)
    (run : Continues module spec (setLocal env name (.scalar (.int next)))
      (body ++ [Calls.loopContinuation body (.rangeStepLoop name (next + stride) stop stride body otherwise)]) heap post globals) :
    Continues module spec env [.rangeStepLoop name next stop stride body otherwise] heap post globals := by
  intro following anchors stack
  apply RunsRel.next (following := ⟨.block (body ++ [Calls.loopContinuation body
    (.rangeStepLoop name (next + stride) stop stride body otherwise)] ++ following)
      (setLocal env name (.scalar (.int next))), stack, heap⟩)
    (by trivial) (by simp [HeapCalls.step, active])
  simpa only [List.append_assoc] using run following anchors stack

theorem range_done (finished : Calls.rangeContinues next stop stride = false)
    (run : Continues module spec env otherwise heap post globals) :
    Continues module spec env [.rangeStepLoop name next stop stride body otherwise] heap post globals := by
  intro following anchors stack
  exact RunsRel.next (by trivial) (by simp [HeapCalls.step, finished]) (run following anchors stack)

end Continues

/-- Each callback retains the invariant while updating the mathematical fold state. -/
theorem range_fold {State : Type u} (name : String) (body : List Stmt)
    (noControl : Calls.loopControl body = false) (update : State → Nat → State)
    (Invariant : State → HeapCalls.Env → Mutable.Heap → Prop) (lower upper : Nat)
    (bindInvariant : ∀ state env heap index, Invariant state env heap → lower ≤ index → index < upper →
      Invariant state (setLocal env name (.scalar (.int index))) heap)
    (cell : ∀ state env heap index, Invariant state env heap → lower ≤ index → index < upper →
      env name = some (.scalar (.int index)) →
      Continues module spec env body heap (Invariant (update state index)) globals)
    (count first : Nat) (start : lower ≤ first) (stop : first + count = upper)
    (state : State) (env : HeapCalls.Env) (heap : Mutable.Heap) (safe : Invariant state env heap) :
    Continues module spec env [.rangeStepLoop name first upper 1 body []] heap
      (Invariant ((List.range' first count).foldl update state)) globals := by
  induction count generalizing first state env heap with
  | zero =>
      apply Continues.range_done (by simp [Calls.rangeContinues]; omega)
      exact Continues.nil safe
  | succ count ih =>
      let entered := setLocal env name (.scalar (.int first))
      have bound := bindInvariant state env heap first safe start (by omega)
      have ran := cell state entered heap first bound start (by omega) (by simp [entered, setLocal])
      have continued := ran.append (rest := [.rangeStepLoop name (first + 1) upper 1 body []])
        (fun middleEnv middle middleSafe => ih (first + 1) (by omega) (by omega) (update state first) middleEnv middle middleSafe)
      apply Continues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, noControl, Bool.false_eq_true, if_false,
        Int.natCast_add, Int.natCast_one, List.range'_succ, List.foldl_cons] using continued

end ProjectEulerSolutions.HeapMemoBlock
