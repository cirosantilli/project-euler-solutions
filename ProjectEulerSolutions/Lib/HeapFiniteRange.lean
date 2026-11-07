import ProjectEulerSolutions.Lib.HeapCollection
import Mathlib.Data.List.FinRange

/-! Structural stepped-range execution with callbacks indexed by the finite iteration domain. -/
namespace ProjectEulerSolutions.HeapFiniteRange
open FormalSW.PyAst HeapCalls
set_option Elab.async false
universe u

theorem loop {State : Type u} (program : Module) (name : String) (body : List Stmt)
    (noControl : Calls.loopControl body = false) (width start stride : Nat) (positive : 0 < stride)
    (storage : State → Mutable.Heap) (update : State → Fin width → State)
    (invariant : HeapCalls.Env → Prop) (globals : Mutable.Globals)
    (bindInvariant : ∀ env i, invariant env →
      invariant (setLocal env name (.scalar (.int (start + stride * (i : Fin width).val)))))
    (callback : ∀ state env i, invariant env →
      env name = some (.scalar (.int (start + stride * (i : Fin width).val))) →
      ∃ afterEnv, BlockContinues program env body (storage state) afterEnv (storage (update state i)) globals ∧
        invariant afterEnv)
    (remaining : List (Fin width)) (position : Nat) (suffix : (List.finRange width).drop position = remaining)
    (within : position ≤ width) (state : State) (env : HeapCalls.Env) (valid : invariant env) :
    ∃ afterEnv, BlockContinues program env
      [.rangeStepLoop name (start + stride * position) (start + stride * width) stride body []]
      (storage state) afterEnv (storage (remaining.foldl update state)) globals ∧ invariant afterEnv := by
  induction remaining generalizing position state env with
  | nil =>
      have length := congrArg List.length suffix
      simp only [List.length_drop, List.length_finRange, List.length_nil] at length
      have atEnd : position = width := by omega
      subst position
      refine ⟨env, BlockContinues.range_done ?_ .nil, valid⟩
      simp [Calls.rangeContinues]
  | cons i rest ih =>
      have item : (List.finRange width)[position]? = some i := by
        simpa [suffix] using (List.head?_drop (l := List.finRange width) (i := position)).symm
      obtain ⟨bound, equal⟩ := List.getElem?_eq_some_iff.mp item
      have value : position = i.val := by
        simpa only [List.getElem_finRange, Fin.val_cast, Fin.val_mk] using congrArg Fin.val equal
      subst position
      let entered := setLocal env name (.scalar (.int (start + stride * i.val)))
      obtain ⟨middleEnv, executed, middleValid⟩ := callback state entered i
        (bindInvariant env i valid) (by simp [entered, setLocal])
      obtain ⟨afterEnv, continued, afterValid⟩ := ih (i.val + 1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) (by omega) (update state i) middleEnv middleValid
      refine ⟨afterEnv, ?_, afterValid⟩
      apply BlockContinues.range_step
      · have less : start + stride * i.val < start + stride * width :=
          Nat.add_lt_add_left (Nat.mul_lt_mul_of_pos_left i.isLt positive) start
        simp [Calls.rangeContinues]
        omega
      · simpa only [Calls.loopContinuation, noControl, Bool.false_eq_true, if_false, List.foldl_cons,
          Nat.mul_add, Nat.mul_one, Nat.add_assoc, Int.natCast_add, Int.natCast_one,
          Int.mul_add, Int.mul_one, Int.add_assoc] using executed.append continued

end ProjectEulerSolutions.HeapFiniteRange
