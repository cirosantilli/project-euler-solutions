import ProjectEulerSolutions.Lib.HeapFiniteRange
import lang.py.FormalSW.TruthCalls

/-! Finite lazy searches return at the first match, including inside nested ranges. -/
namespace ProjectEulerSolutions.HeapFiniteAny
open FormalSW.PyAst HeapCalls
set_option Elab.async false

/-- A matching iteration returns from the generator; otherwise its locals satisfy the invariant. -/
def Scans (program : Module) (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (invariant : HeapCalls.Env → Prop) (matched : Bool) (globals : Mutable.Globals) : Prop :=
  if matched then
    ∀ following, BlockReturns program env (body ++ following) heap (.scalar (.bool true)) heap globals
  else
    ∃ afterEnv, BlockContinues program env body heap afterEnv heap globals ∧ invariant afterEnv

theorem Scans.weaken {program : Module} {env : HeapCalls.Env} {body : List Stmt} {heap : Mutable.Heap}
    {invariant weaker : HeapCalls.Env → Prop} {matched : Bool} {globals : Mutable.Globals}
    (scanned : Scans program env body heap invariant matched globals)
    (implies : ∀ scope, invariant scope → weaker scope) :
    Scans program env body heap weaker matched globals := by
  cases matched with
  | false =>
      obtain ⟨afterEnv, continued, valid⟩ := scanned
      exact ⟨afterEnv, continued, implies afterEnv valid⟩
  | true => exact scanned

theorem test (program : Module) (env : HeapCalls.Env) (element : Expr) (heap : Mutable.Heap)
    (invariant : HeapCalls.Env → Prop) (matched : Bool) (globals : Mutable.Globals)
    (valid : invariant env)
    (evaluated : Evaluates program env element heap (.scalar (.bool matched)) heap globals) :
    Scans program env (truthBody false element) heap invariant matched globals := by
  cases matched with
  | false => exact ⟨env, BlockContinues.branch evaluated rfl .nil, valid⟩
  | true =>
      intro following
      exact BlockReturns.branch evaluated rfl (BlockReturns.return_value Evaluates.constant)

theorem finish {program : Module} {env : HeapCalls.Env} {body : List Stmt} {heap : Mutable.Heap}
    {invariant : HeapCalls.Env → Prop} {matched : Bool} {globals : Mutable.Globals}
    (scanned : Scans program env body heap invariant matched globals) :
    BlockReturns program env (body ++ [.return (.constant (.bool false))])
      heap (.scalar (.bool matched)) heap globals := by
  cases matched with
  | false =>
      obtain ⟨afterEnv, continued, _⟩ := scanned
      exact continued.returns (BlockReturns.return_value Evaluates.constant)
  | true => exact scanned _

theorem for_range {program : Module} {env : HeapCalls.Env} {body : List Stmt} {heap : Mutable.Heap}
    {invariant : HeapCalls.Env → Prop} {matched : Bool} {globals : Mutable.Globals}
    {name : String} {expressions : List Expr} {values : List Mutable.Value} {start stop stride : Int}
    (arguments : Evaluates.Arguments program env expressions heap values heap globals)
    (bounds : rangeBounds values = .ok (start, stop, stride))
    (scanned : Scans program env [.rangeStepLoop name start stop stride body []] heap invariant matched globals) :
    Scans program env [.forIn (.name name) (.call (.name "range") expressions) body []]
      heap invariant matched globals := by
  cases matched with
  | false =>
      obtain ⟨afterEnv, continued, valid⟩ := scanned
      exact ⟨afterEnv, BlockContinues.for_range arguments bounds continued, valid⟩
  | true =>
      intro following
      exact BlockReturns.for_range arguments bounds (scanned following)

theorem loop (program : Module) (name : String) (body : List Stmt)
    (noControl : Calls.loopControl body = false) (width start stride : Nat) (positive : 0 < stride)
    (heap : Mutable.Heap) (invariant : HeapCalls.Env → Prop) (predicate : Fin width → Bool)
    (globals : Mutable.Globals)
    (bindInvariant : ∀ env i, invariant env →
      invariant (setLocal env name (.scalar (.int (start + stride * (i : Fin width).val)))))
    (callback : ∀ env i, invariant env →
      env name = some (.scalar (.int (start + stride * (i : Fin width).val))) →
      Scans program env body heap invariant (predicate i) globals)
    (remaining : List (Fin width)) (position : Nat) (suffix : (List.finRange width).drop position = remaining)
    (within : position ≤ width) (env : HeapCalls.Env) (valid : invariant env) :
    Scans program env
      [.rangeStepLoop name (start + stride * position) (start + stride * width) stride body []]
      heap invariant (remaining.any predicate) globals := by
  induction remaining generalizing position env with
  | nil =>
      have length := congrArg List.length suffix
      simp only [List.length_drop, List.length_finRange, List.length_nil] at length
      have atEnd : position = width := by omega
      subst position
      exact ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, valid⟩
  | cons i rest ih =>
      have item : (List.finRange width)[position]? = some i := by
        simpa [suffix] using (List.head?_drop (l := List.finRange width) (i := position)).symm
      obtain ⟨bound, equal⟩ := List.getElem?_eq_some_iff.mp item
      have value : position = i.val := by
        simpa only [List.getElem_finRange, Fin.val_cast, Fin.val_mk] using congrArg Fin.val equal
      subst position
      let entered := setLocal env name (.scalar (.int (start + stride * i.val)))
      have enteredValid : invariant entered := bindInvariant env i valid
      have tested := callback entered i enteredValid (by simp [entered, setLocal])
      have continues : Calls.rangeContinues (start + stride * i.val) (start + stride * width) stride = true := by
        have less := Nat.add_lt_add_left (Nat.mul_lt_mul_of_pos_left i.isLt positive) start
        simp [Calls.rangeContinues]
        omega
      cases found : predicate i with
      | true =>
          simp only [List.any_cons, found, Bool.true_or, Scans, if_true] at tested ⊢
          intro following
          apply BlockReturns.range_step continues
          simpa only [entered, Int.natCast_add, Int.natCast_mul, List.append_assoc, List.nil_append] using tested
            ([Calls.loopContinuation body (.rangeStepLoop name
              (start + stride * i.val + stride) (start + stride * width) stride body [])] ++ following)
      | false =>
          simp only [found, Scans, Bool.false_eq_true, if_false] at tested
          obtain ⟨middleEnv, checked, middleValid⟩ := tested
          have restChecked := ih (i.val + 1) (by simp [List.drop_add_one_eq_tail_drop, suffix])
            (by omega) middleEnv middleValid
          simp only [List.any_cons, found, Bool.false_or]
          cases tailFound : rest.any predicate with
          | true =>
              simp only [tailFound, Scans, if_true] at restChecked ⊢
              intro following
              apply BlockReturns.range_step continues
              simpa only [Calls.loopContinuation, noControl, Bool.false_eq_true, if_false,
                Nat.mul_add, Nat.mul_one, Nat.add_assoc, Int.natCast_add, Int.natCast_one,
                Int.mul_add, Int.mul_one, Int.add_assoc, List.append_assoc] using
                checked.returns (restChecked following)
          | false =>
              simp only [tailFound, Scans, Bool.false_eq_true, if_false] at restChecked ⊢
              obtain ⟨afterEnv, continued, afterValid⟩ := restChecked
              refine ⟨afterEnv, BlockContinues.range_step continues ?_, afterValid⟩
              simpa only [Calls.loopContinuation, noControl, Bool.false_eq_true, if_false,
                Nat.mul_add, Nat.mul_one, Nat.add_assoc, Int.natCast_add, Int.natCast_one,
                Int.mul_add, Int.mul_one, Int.add_assoc] using checked.append continued

end ProjectEulerSolutions.HeapFiniteAny
