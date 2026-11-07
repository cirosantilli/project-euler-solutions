import «.formalsw».solvers.«4.py.ast»
import data.«project-euler-statements».«.formalsw».data.py.project_euler_statements.«p4.py.ast»
import lang.py.FormalSW.CallsCostControl
import lang.py.FormalSW.HeapCallsCostLists
import Mathlib.Tactic.Linarith

/-! Source-linked execution envelopes in the number of decimal digits.
The optimized proof uses Calls; its string conversion, slicing, comparison and
integer arithmetic are atomic interpreter operations. The reference uses
HeapCalls and counts the linked Python digits_le loop. Heap allocation, list
append/reversal/equality and scalar arithmetic remain atomic transitions.
These are transition bounds, not bit complexity or weighted collection costs. -/
namespace solvers.«4».complexity
open FormalSW.PyAst
set_option maxRecDepth 4096
set_option maxHeartbeats 1200000

namespace Optimized
open Calls Calls.Cost
private abbrev program := Solvers.P4Ast.module
private def outerBody : List Stmt :=
  match executableBody ((findFunctionIn program.body "largest_palindrome_product").getD ([], [])).2 with
  | [_, .forIn _ _ body _, _] => body
  | _ => []
private def innerBody : List Stmt :=
  match outerBody with
  | [_, .forIn _ _ body _] => body
  | _ => []
private def palindromeTest : Expr :=
  match innerBody with
  | [_, _, _, .ifThen condition _ _] => condition
  | _ => .constant .none

private structure Bounds (env : ScalarEnv) (lo hi : Int) : Prop where
  lower : env "lo" = some (.int lo)
  upper : env "hi" = some (.int hi)
  best : ∃ value : Int, env "best" = some (.int value)

private theorem Bounds.setAux (h : Bounds env lo hi) (name : String) (value : Constant)
    (hl : "lo" ≠ name) (hh : "hi" ≠ name) (hb : "best" ≠ name) :
    Bounds (env.set name value) lo hi := by
  obtain ⟨best, getBest⟩ := h.best
  exact ⟨by simpa [ScalarEnv.set, hl] using h.lower,
    by simpa [ScalarEnv.set, hh] using h.upper,
    best, by simpa [ScalarEnv.set, hb] using getBest⟩

private theorem Bounds.setBest (h : Bounds env lo hi) (value : Int) :
    Bounds (env.set "best" (.int value)) lo hi :=
  ⟨by simpa [ScalarEnv.set] using h.lower,
    by simpa [ScalarEnv.set] using h.upper, value, by simp [ScalarEnv.set]⟩

private theorem str_cost (env : ScalarEnv) (value : Int) (found : env "prod" = some (.int value)) :
    EvaluatesIn program env (.builtinCall .str [.name "prod"]) 4 (.string (toString value)) := by
  intro stack
  exact .next rfl ((EvaluatesIn.lookup found _).trans (.next rfl (.next rfl .refl)))

private theorem palindrome_cost (env : ScalarEnv) (s : String) (found : env "s" = some (.string s)) :
    EvaluatesIn program env palindromeTest 16 (.bool (decide (s = String.ofList s.toList.reverse))) := by
  apply EvaluatesIn.compare (ca := 1) (cb := 12) (bv := .string (String.ofList s.toList.reverse)) (EvaluatesIn.lookup found)
  · intro stack
    have tail : ReachesIn (step program)
        ⟨.primitive .slice [.string s, .none, .none, .int (-1)], stack⟩ 1
        ⟨.value (.string (String.ofList s.toList.reverse)), stack⟩ :=
      .next (by simp [step, outcome, Primitive.apply, Strings.slice_reverse]) .refl
    refine .next rfl ?_
    change ReachesIn (step program) ⟨.expr (.name "s") env,
      .primitiveArguments .slice [] [.constant .none, .constant .none, .unaryOp .uSub (.constant (.int 1))] env :: stack⟩ 11 _
    apply (EvaluatesIn.lookup (module := program) found _).trans (second := 10)
    repeat first | exact tail | refine ReachesIn.next rfl ?_
  · simp [scalarCompare]

private theorem inner_loop (remaining : Nat) (b stop a lo hi : Int) (endpoint : b = stop + remaining)
    (env : ScalarEnv) (following : List Stmt) (budget : Nat)
    (bounds : Bounds env lo hi) (getA : env "a" = some (.int a))
    (after : ∀ nextEnv, Bounds nextEnv lo hi → BlockWithin program nextEnv following budget) :
    BlockWithin program env (.rangeStepLoop "b" b stop (-1) innerBody [] :: following)
      (44 * remaining + 1 + budget) := by
  induction remaining generalizing b env with
  | zero =>
      simpa [Nat.add_comm] using BlockWithin.range_done (name := "b") (next := b) (stop := stop) (stride := -1)
        (body := innerBody) (otherwise := []) (by simp [rangeContinues]; omega) (after env bounds)
  | succ remaining ih =>
      obtain ⟨best, getBest⟩ := bounds.best
      refine (BlockWithin.range_step (bound := 7 + (44 * remaining + 36 + budget)) (by simp [rangeContinues]; omega) ?_).mono ?_
      · let envB := env.set "b" (.int b)
        have boundsB : Bounds envB lo hi := bounds.setAux "b" (.int b) (by decide) (by decide) (by decide)
        refine BlockWithin.assign (cost := 5) (bound := 44 * remaining + 36 + budget) (value := .int (a * b)) ?_ ?_
        · exact EvaluatesIn.binary (EvaluatesIn.lookup (by simpa [envB, ScalarEnv.set] using getA))
            (EvaluatesIn.lookup rfl) rfl
        · let envP := envB.set "prod" (.int (a * b))
          have boundsP : Bounds envP lo hi := boundsB.setAux "prod" (.int (a * b)) (by decide) (by decide) (by decide)
          have guard : EvaluatesIn program envP (.compare (.name "prod") [(.ltE, .name "best")]) 5
              (.bool (decide (a * b ≤ best))) :=
            EvaluatesIn.compare (EvaluatesIn.lookup rfl)
              (EvaluatesIn.lookup (by simpa [envP, envB, ScalarEnv.set] using getBest))
              (by simp [scalarCompare, evalCmpOp])
          by_cases beaten : a * b ≤ best
          · refine (BlockWithin.branch (bound := budget + 1) (by simpa [beaten] using guard) ?_).mono (large := 44 * remaining + 36 + budget) ?_
            · exact BlockWithin.break_loop rfl (after envP boundsP)
            · omega
          · refine (BlockWithin.branch (bound := 6 + (16 + 2 + (44 * remaining + 5 + budget)))
              (by simpa [beaten] using guard) ?_).mono (large := 44 * remaining + 36 + budget) ?_
            · refine BlockWithin.assign (bound := 16 + 2 + (44 * remaining + 5 + budget)) (str_cost envP (a * b) rfl) ?_
              let envS := envP.set "s" (.string (toString (a * b)))
              have boundsS : Bounds envS lo hi := boundsP.setAux "s" (.string (toString (a * b)))
                (by decide) (by decide) (by decide)
              refine BlockWithin.branch (bound := 44 * remaining + 5 + budget) (palindrome_cost envS (toString (a * b)) rfl) ?_
              by_cases palindrome : toString (a * b) = String.ofList (toString (a * b)).toList.reverse
              · have test : decide (toString (a * b) = String.ofList (toString (a * b)).toList.reverse) = true := decide_eq_true palindrome
                simp only [test, ↓reduceIte]
                refine (BlockWithin.assign (EvaluatesIn.lookup rfl)
                  (BlockWithin.break_loop rfl (after _ (boundsS.setBest (a * b))))).mono (large := 44 * remaining + 5 + budget) ?_
                omega
              · have test : decide (toString (a * b) = String.ofList (toString (a * b)).toList.reverse) = false := decide_eq_false palindrome
                simp only [test, Bool.false_eq_true, ↓reduceIte]
                have next := ih (b - 1) (by omega) envS boundsS
                  (by simpa [envS, envP, envB, ScalarEnv.set] using getA)
                have resumed := BlockWithin.resume next
                apply BlockWithin.mono (small := (44 * remaining + 1 + budget) + 1)
                · simpa only [show b + (-1) = b - 1 by omega] using resumed
                · omega
            · omega
      · omega

private theorem finish (bounds : Bounds env lo hi) :
    BlockWithin program env [.return (.name "best")] 2 := by
  obtain ⟨best, found⟩ := bounds.best
  exact BlockWithin.return_value (EvaluatesIn.lookup found)

private theorem outer_loop (remaining width : Nat) (a lo hi : Int)
    (endpoint : a = lo - 1 + remaining) (small : remaining ≤ width)
    (env : ScalarEnv) (bounds : Bounds env lo hi) :
    BlockWithin program env [.rangeStepLoop "a" a (lo - 1) (-1) outerBody [], .return (.name "best")]
      ((44 * width + 30) * remaining + 3) := by
  induction remaining generalizing a env with
  | zero =>
      simpa using BlockWithin.range_done (by simp [rangeContinues]; omega) (finish bounds)
  | succ remaining ih =>
      obtain ⟨best, getBest⟩ := bounds.best
      refine (BlockWithin.range_step (bound := 44 * width + 29 + ((44 * width + 30) * remaining + 3)) (by simp [rangeContinues]; omega) ?_).mono ?_
      · let envA := env.set "a" (.int a)
        have boundsA : Bounds envA lo hi := bounds.setAux "a" (.int a) (by decide) (by decide) (by decide)
        have guard : EvaluatesIn program envA
            (.compare (.binOp (.name "a") .mult (.name "hi")) [(.lt, .name "best")]) 9
            (.bool (decide (a * hi < best))) :=
          EvaluatesIn.compare (EvaluatesIn.binary (EvaluatesIn.lookup rfl) (EvaluatesIn.lookup boundsA.upper) rfl)
            (EvaluatesIn.lookup (by simpa [envA, ScalarEnv.set] using getBest))
            (by simp [scalarCompare, evalCmpOp])
        by_cases beaten : a * hi < best
        · refine (BlockWithin.branch (by simpa [beaten] using guard)
            (BlockWithin.break_loop rfl (finish boundsA))).mono
            (large := 44 * width + 29 + ((44 * width + 30) * remaining + 3)) ?_
          omega
        · refine (BlockWithin.branch (bound := 12 + 2 + (44 * (remaining + 1) + 1 + (((44 * width + 30) * remaining + 3) + 1)))
            (by simpa [beaten] using guard) ?_).mono
            (large := 44 * width + 29 + ((44 * width + 30) * remaining + 3)) ?_
          · refine BlockWithin.for_range (values := [.int a, .int (lo - 1), .int (-1)]) ?_ rfl ?_
            · exact .cons (EvaluatesIn.lookup rfl) (.cons
                (EvaluatesIn.binary (EvaluatesIn.lookup boundsA.lower) EvaluatesIn.constant rfl)
                (.cons (EvaluatesIn.unary EvaluatesIn.constant rfl) .nil))
            · apply inner_loop (remaining + 1) a (lo - 1) a lo hi endpoint envA _
                (((44 * width + 30) * remaining + 3) + 1) boundsA rfl
              intro nextEnv nextBounds
              have next := ih (a - 1) (by omega) (by omega) nextEnv nextBounds
              simpa only [show a + (-1) = a - 1 by omega] using BlockWithin.resume next
          · have grow := Nat.mul_le_mul_left 44 small
            omega
      · rw [Nat.mul_succ]
        have grow := Nat.mul_le_mul_left 44 small
        omega

private theorem largest_cost (lo hi : Int) (width : Nat) (endpoint : hi = lo - 1 + width) :
    ∃ cost value, cost ≤ (44 * width + 30) * width + 21 ∧ ∀ stack,
      ReachesIn (step program) ⟨.invoke "largest_palindrome_product" [.int lo, .int hi], stack⟩ cost
        ⟨.value value, stack⟩ := by
  let env := argumentEnv [⟨"lo", some (.name "int"), none⟩, ⟨"hi", some (.name "int"), none⟩] [.int lo, .int hi]
  have loops := outer_loop width width hi lo hi endpoint (Nat.le_refl _) (env.set "best" (.int 0))
    ⟨rfl, rfl, 0, rfl⟩
  have range := BlockWithin.for_range (name := "a") (expressions := [.name "hi",
      .binOp (.name "lo") .sub (.constant (.int 1)), .unaryOp .uSub (.constant (.int 1))])
    (.cons (EvaluatesIn.lookup (value := .int hi) rfl)
      (.cons (EvaluatesIn.binary (EvaluatesIn.lookup (value := .int lo) rfl) EvaluatesIn.constant rfl)
        (.cons (EvaluatesIn.unary EvaluatesIn.constant rfl) .nil))) rfl loops
  have block := BlockWithin.assign (name := "best") (EvaluatesIn.constant (env := env)) range
  obtain ⟨cost, value, bounded, run⟩ := block
  exact ⟨cost + 1, value, by omega, fun stack => .next rfl (run stack)⟩

/-- The optimized source, including solve's numeric bounds and its helper call. -/
theorem source_bound (n : Nat) (positive : 0 < n) :
    ∃ cost value, FinishesIn (step Solvers.P4Ast.module) ⟨.invoke "solve" [.int n], []⟩ cost (.ok value) ∧
      cost ≤ 150 * 10 ^ (2 * n) := by
  let lo : Nat := 10 ^ (n - 1)
  let hi : Nat := 10 ^ n - 1
  have powerPositive : 1 ≤ (10 : Nat) ^ n := Nat.one_le_pow _ _ (by decide)
  have lowerBound : lo ≤ hi + 1 := by
    dsimp [lo, hi]
    rw [Nat.sub_add_cancel powerPositive]
    exact Nat.pow_le_pow_right (by decide) (Nat.sub_le n 1)
  let width : Nat := hi + 1 - lo
  have endpoint : (hi : Int) = (lo : Int) - 1 + width := by omega
  obtain ⟨helperCost, value, bounded, helper⟩ := largest_cost lo hi width endpoint
  let env := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]
  let envLo := env.set "lo" (.int lo)
  let envHi := envLo.set "hi" (.int hi)
  let source := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
  have lower : EvaluatesIn program env
      (.ifExp (.compare (.name "n") [(.eq, .constant (.int 0))]) (.constant (.int 0))
        (.binOp (.constant (.int 10)) .pow (.binOp (.name "n") .sub (.constant (.int 1))))) 16 (.int lo) := by
    apply EvaluatesIn.conditional (ct := 5) (cb := 9) (flag := false)
    · exact EvaluatesIn.compare (EvaluatesIn.lookup rfl) EvaluatesIn.constant
        (by simp [scalarCompare, evalCmpOp]; omega)
    · apply EvaluatesIn.binary (ca := 1) (cb := 5) (bv := .int ((n - 1 : Nat) : Int)) EvaluatesIn.constant
      · exact EvaluatesIn.binary (EvaluatesIn.lookup rfl) EvaluatesIn.constant (by simp [scalarBinary]; omega)
      · simp [scalarBinary, lo]
  have upper : EvaluatesIn program envLo
      (.binOp (.binOp (.constant (.int 10)) .pow (.name "n")) .sub (.constant (.int 1))) 9 (.int hi) := by
    apply EvaluatesIn.binary (ca := 5) (cb := 1) (av := .int ((10 ^ n : Nat) : Int))
    · exact EvaluatesIn.binary EvaluatesIn.constant (EvaluatesIn.lookup rfl) (by simp [scalarBinary])
    · exact EvaluatesIn.constant
    · simp [scalarBinary, hi, Int.natCast_sub powerPositive]
  have start : ReachesIn (step program) ⟨.invoke "solve" [.int n], []⟩ 1 ⟨.block source env, []⟩ := .next rfl .refl
  have tail : ReachesIn (step program) ⟨.block (source.drop 2) envHi, []⟩ 6
      ⟨.invoke "largest_palindrome_product" [.int lo, .int hi], []⟩ := by
    repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_
  have lead : ReachesIn (step program) ⟨.invoke "solve" [.int n], []⟩ 36
      ⟨.invoke "largest_palindrome_product" [.int lo, .int hi], []⟩ :=
    ((start.trans (Cost.assign lower "lo" (source.drop 1) [])).trans
      (Cost.assign upper "hi" (source.drop 2) [])).trans tail
  refine ⟨36 + helperCost + 1, value, (lead.trans (helper [])).finishes (.done rfl), ?_⟩
  have widthBound : width ≤ 10 ^ n := by
    dsimp [width]
    exact (Nat.sub_le (hi + 1) lo).trans (by dsimp [hi]; rw [Nat.sub_add_cancel powerPositive])
  have square : 10 ^ (2 * n) = (10 ^ n) * 10 ^ n := by rw [two_mul, pow_add]
  rw [square]
  nlinarith only [bounded, powerPositive, widthBound, Nat.mul_self_le_mul_self widthBound]

end Optimized
namespace Reference
open HeapCalls HeapCalls.Cost
private abbrev program := P4Ast.module
private def encode (items : List Int) : List Mutable.Value := items.map (fun i => .scalar (.int i))
private def digitsLoop : List Stmt :=
  (executableBody ((findFunctionIn program.body "digits_le").getD ([], [])).2).drop 2
private def digitsBody : List Stmt :=
  match digitsLoop with
  | .whileDo _ body _ :: _ => body
  | _ => []

/-- Every source digit-loop iteration divides its natural input by ten. -/
private theorem digits_loop (k m : Nat) (budget : m < 10 ^ k) (env : HeapCalls.Env)
    (prior : Heap) (acc : List Int)
    (getN : env "n" = some (.scalar (.int m)))
    (getBase : env "base" = some (.scalar (.int 10)))
    (getOut : env "out" = some (.ref prior.length)) :
    ∃ cost items, cost ≤ 21 * k + 5 ∧ BlockReturnsIn program env digitsLoop
      (prior ++ [.list (encode acc)]) cost (.ref prior.length) (prior ++ [.list (encode items)]) := by
  induction k generalizing m env acc with
  | zero =>
      have zero : m = 0 := by simpa using budget
      subst m
      refine ⟨5, acc, by omega, ?_⟩
      apply Lists.whileLoop (cc := 1) (cb := 2) (flag := false) (EvaluatesIn.lookup getN)
      · rfl
      · exact BlockReturnsIn.return_value (EvaluatesIn.lookup getOut)
  | succ k ih =>
      by_cases zero : m = 0
      · subst m
        refine ⟨5, acc, by omega, ?_⟩
        apply Lists.whileLoop (cc := 1) (cb := 2) (flag := false) (EvaluatesIn.lookup getN)
        · rfl
        · exact BlockReturnsIn.return_value (EvaluatesIn.lookup getOut)
      · have smaller : m / 10 < 10 ^ k := by
          apply (Nat.div_lt_iff_lt_mul (by decide : 0 < 10)).mpr
          simpa [Nat.pow_succ] using budget
        let updated := setLocal env "n" (.scalar (.int (m / 10)))
        obtain ⟨cost, items, bounded, tail⟩ := ih (m / 10) smaller updated (acc ++ [((m % 10 : Nat) : Int)])
          (by simp [updated, setLocal]) (by simpa [updated, setLocal] using getBase)
          (by simpa [updated, setLocal] using getOut)
        have appended : BlockContinuesIn program env [digitsBody[0]]
            (prior ++ [.list (encode acc)]) 12 env (prior ++ [.list (encode (acc ++ [((m % 10 : Nat) : Int)]))]) := by
          apply BlockContinuesIn.discard
          have item : EvaluatesIn program env (.binOp (.name "n") .mod (.name "base"))
              (prior ++ [.list (encode acc)]) 5 (.scalar (.int (m % 10))) (prior ++ [.list (encode acc)]) :=
            EvaluatesIn.binary (EvaluatesIn.lookup getN) (EvaluatesIn.lookup getBase)
              (by simp [Mutable.binary, scalarBinary])
          simpa [encode] using Lists.append (old := encode acc) (current := encode acc) (EvaluatesIn.lookup getOut) (by simp) item (by simp)
        have divided : BlockContinuesIn program env [digitsBody[1]]
            (prior ++ [.list (encode (acc ++ [((m % 10 : Nat) : Int)]))]) 6 updated
            (prior ++ [.list (encode (acc ++ [((m % 10 : Nat) : Int)]))]) := by
          apply BlockContinuesIn.augAssign getN (EvaluatesIn.lookup getBase)
          simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        refine ⟨1 + (18 + cost) + 2, items, by omega, ?_⟩
        apply Lists.whileLoop (cc := 1) (cb := 18 + cost) (flag := true) (EvaluatesIn.lookup getN)
        · simp [Mutable.truth, scalarTruth, zero]
        · exact (appended.append divided).returns tail

private theorem digits_cost (m k : Nat) (budget : m < 10 ^ k) (heap : Heap) :
    ∃ cost items, cost ≤ 21 * k + 17 ∧ ReturnsIn program "digits_le" [.scalar (.int m)] heap cost
      (.ref heap.length) (heap ++ [.list (encode items)]) := by
  let env := functionEnv ((findFunctionIn program.body "digits_le").getD ([], [])).1
    ((findFunctionIn program.body "digits_le").getD ([], [])).2 [.scalar (.int m), .scalar (.int 10)] []
  have getN : env "n" = some (.scalar (.int m)) := rfl
  have test : EvaluatesIn program env (.compare (.name "n") [(.eq, .constant (.int 0))])
      heap 5 (.scalar (.bool (decide (m = 0)))) heap :=
    EvaluatesIn.compare (EvaluatesIn.lookup getN) EvaluatesIn.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases zero : m = 0
  · refine ⟨11, [], by omega, ?_⟩
    apply BlockReturnsIn.function rfl rfl
    apply Lists.branch (cc := 5) (cb := 3) (flag := true) (value := .scalar (.bool true))
    · simpa only [show decide (m = 0) = true from decide_eq_true zero] using test
    · rfl
    · exact BlockReturnsIn.return_value (Lists.list .nil)
  · obtain ⟨cost, items, bounded, tail⟩ := digits_loop k m budget (setLocal env "out" (.ref heap.length)) heap []
        (by simpa [setLocal] using getN) rfl rfl
    refine ⟨(5 + (2 + cost + 2) + 2) + 1, items, by omega, ?_⟩
    apply BlockReturnsIn.function rfl rfl
    apply Lists.branch (cc := 5) (cb := 2 + cost + 2) (flag := false) (value := .scalar (.bool false))
    · simpa only [show decide (m = 0) = false from decide_eq_false zero] using test
    · rfl
    · exact BlockReturnsIn.assign (Lists.list .nil) tail

private def outerBody : List Stmt :=
  match executableBody ((findFunctionIn program.body "naive").getD ([], [])).2 with
  | [_, _, _, .forIn _ _ body _, _] => body
  | _ => []
private def innerBody : List Stmt :=
  match outerBody with
  | [.forIn _ _ body _] => body
  | _ => []

private structure Invariant (env : HeapCalls.Env) (hi : Nat) : Prop where
  upper : env "hi" = some (.scalar (.int hi))
  best : ∃ best : Nat, env "max_" = some (.scalar (.int best))

private theorem Invariant.setAux (h : Invariant env hi) (name : String) (value : Mutable.Value)
    (hh : "hi" ≠ name) (hm : "max_" ≠ name) : Invariant (setLocal env name value) hi := by
  obtain ⟨best, found⟩ := h.best
  exact ⟨by simpa [setLocal, hh] using h.upper, best, by simpa [setLocal, hm] using found⟩

private theorem Invariant.setBest (h : Invariant env hi) (best : Nat) :
    Invariant (setLocal env "max_" (.scalar (.int best))) hi :=
  ⟨by simpa [setLocal] using h.upper, best, by simp [setLocal]⟩

private theorem body_cost (n a b hi : Nat) (productBound : a * b < 10 ^ (2 * n))
    (env : HeapCalls.Env) (heap : Heap) (invariant : Invariant env hi)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b))) :
    ∃ cost finalEnv finalHeap, cost ≤ 42 * n + 50 ∧
      BlockContinuesIn program env innerBody heap cost finalEnv finalHeap ∧
      Invariant finalEnv hi ∧ finalEnv "a" = some (.scalar (.int a)) := by
  obtain ⟨best, getBest⟩ := invariant.best
  obtain ⟨digitsCost, items, digitsBound, digitsRun⟩ := digits_cost (a * b) (2 * n) productBound heap
  let envP := setLocal env "prod" (.scalar (.int ((a * b : Nat) : Int)))
  let envD := setLocal envP "ds" (.ref heap.length)
  let heapD := heap ++ [.list (encode items)]
  let heapR := heapD ++ [.list (encode items.reverse)]
  have assigned : BlockContinuesIn program env [innerBody[0]] heap 7 envP heap :=
    BlockContinuesIn.assign (EvaluatesIn.binary (EvaluatesIn.lookup ha) (EvaluatesIn.lookup hb)
      (by simp [Mutable.binary, scalarBinary]))
  have digits : BlockContinuesIn program envP [innerBody[1]] heap (2 + digitsCost + 1 + 2) envD heapD :=
    BlockContinuesIn.assign (EvaluatesIn.call (.cons (EvaluatesIn.lookup rfl) .nil) digitsRun)
  have test : EvaluatesIn program envD (.compare (.name "ds") [(.eq, .reversedList (.name "ds"))])
      heapD 8 (.scalar (.bool (decide (items = items.reverse)))) heapR := by
    apply EvaluatesIn.compare (ca := 1) (cb := 4) (EvaluatesIn.lookup rfl)
      (Lists.reversed (items := encode items) (EvaluatesIn.lookup rfl) (by simp [heapD]))
    have result := HeapCalls.compare_int_lists items items.reverse (heap := heapR)
      (a := heap.length) (b := heapD.length)
      (by simp [heapR, heapD, encode]) (by simp [heapR, heapD, encode])
    simpa [heapR, encode, List.map_reverse] using result
  have invariantD : Invariant envD hi :=
    (invariant.setAux "prod" _ (by decide) (by decide)).setAux "ds" _ (by decide) (by decide)
  have getA : envD "a" = some (.scalar (.int a)) := by simpa [envD, envP, setLocal] using ha
  by_cases palindrome : items = items.reverse
  · let finalEnv := setLocal envD "max_" (.scalar (.int ((max (a * b) best : Nat) : Int)))
    have updated : BlockContinuesIn program envD [innerBody[2]] heapD (8 + 8 + 2) finalEnv heapR := by
      apply BlockContinuesIn.branch (cc := 8) (cb := 8) (flag := true) (value := .scalar (.bool true))
      · simpa only [show decide (items = items.reverse) = true from decide_eq_true palindrome] using test
      · rfl
      · apply BlockContinuesIn.assign (cost := 6)
        apply EvaluatesIn.builtin (.cons (EvaluatesIn.lookup (value := .scalar (.int ((a * b : Nat) : Int))) (by simp [envD, envP, setLocal]))
          (.cons (EvaluatesIn.lookup (by simpa [envD, envP, setLocal] using getBest)) .nil))
        simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom, HeapCalls.scalarResult,
          scalarCompare, evalCmpOp, heapR]
        split <;> rename_i selected <;> simp [max_def] <;> omega
    refine ⟨7 + (2 + digitsCost + 1 + 2) + (8 + 8 + 2), finalEnv, heapR, by omega, ?_, invariantD.setBest _, ?_⟩
    · exact (assigned.append digits).append updated
    · simpa [finalEnv, setLocal] using getA
  · have skipped : BlockContinuesIn program envD [innerBody[2]] heapD (8 + 0 + 2) envD heapR := by
      apply BlockContinuesIn.branch (cc := 8) (cb := 0) (flag := false) (value := .scalar (.bool false))
      · simpa only [show decide (items = items.reverse) = false from decide_eq_false palindrome] using test
      · rfl
      · exact BlockContinuesIn.nil
    exact ⟨7 + (2 + digitsCost + 1 + 2) + (8 + 0 + 2), envD, heapR, by omega, (assigned.append digits).append skipped, invariantD, getA⟩

private theorem inner_loop (n remaining start hi a : Nat)
    (endpoint : hi + 1 = start + remaining) (aBound : a < 10 ^ n) (hiBound : hi < 10 ^ n)
    (env : HeapCalls.Env) (heap : Heap) (invariant : Invariant env hi)
    (getA : env "a" = some (.scalar (.int a))) :
    ∃ cost finalEnv finalHeap, cost ≤ (42 * n + 51) * remaining + 1 ∧
      BlockContinuesIn program env [.rangeStepLoop "b" start (hi + 1) 1 innerBody []] heap cost finalEnv finalHeap ∧
      Invariant finalEnv hi ∧ finalEnv "a" = some (.scalar (.int a)) := by
  induction remaining generalizing start env heap with
  | zero =>
      refine ⟨1, env, heap, by omega, ?_, invariant, getA⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
  | succ remaining ih =>
      let envB := setLocal env "b" (.scalar (.int start))
      have productBound : a * start < 10 ^ (2 * n) := by
        rw [two_mul, pow_add]
        exact Nat.mul_lt_mul_of_lt_of_lt aBound (by omega)
      obtain ⟨bodyCost, rowEnv, rowHeap, bodyBound, bodyRun, rowInv, rowA⟩ := body_cost n a start hi productBound envB heap
        (invariant.setAux "b" _ (by decide) (by decide))
        (by simpa [envB, setLocal] using getA) rfl
      obtain ⟨cost, finalEnv, finalHeap, bounded, tail, finalInv, finalA⟩ :=
        ih (start + 1) (by omega) rowEnv rowHeap rowInv rowA
      refine ⟨bodyCost + cost + 1, finalEnv, finalHeap, ?_, ?_, finalInv, finalA⟩
      · rw [Nat.mul_succ]; omega
      · apply BlockContinuesIn.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, innerBody, Int.natCast_add] using bodyRun.append tail

private theorem outer_loop (n remaining start hi : Nat) (endpoint : hi + 1 = start + remaining)
    (hiBound : hi < 10 ^ n) (env : HeapCalls.Env) (heap : Heap) (invariant : Invariant env hi) :
    ∃ cost finalEnv finalHeap, cost ≤ ((42 * n + 51) * (hi + 1) + 12) * remaining + 1 ∧
      BlockContinuesIn program env [.rangeStepLoop "a" start (hi + 1) 1 outerBody []] heap cost finalEnv finalHeap ∧
      Invariant finalEnv hi := by
  induction remaining generalizing start env heap with
  | zero =>
      refine ⟨1, env, heap, by omega, ?_, invariant⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
  | succ remaining ih =>
      let envA := setLocal env "a" (.scalar (.int start))
      have invA := invariant.setAux "a" (.scalar (.int start)) (by decide) (by decide)
      obtain ⟨rowCost, rowEnv, rowHeap, rowBound, rowRun, rowInv, _⟩ :=
        inner_loop n (hi + 1 - start) start hi start (by omega) (by omega) hiBound envA heap invA rfl
      have row : BlockContinuesIn program envA outerBody heap (8 + (rowCost + 1) + 1) rowEnv rowHeap := by
        apply BlockContinuesIn.for_range (middle := heap) (values := [.scalar (.int start), .scalar (.int ((hi + 1 : Nat) : Int))])
        · exact .cons (EvaluatesIn.lookup rfl) (.cons
            (EvaluatesIn.binary (EvaluatesIn.lookup invA.upper) EvaluatesIn.constant (by simp [Mutable.binary, scalarBinary])) .nil)
        · rfl
        · exact rowRun
      obtain ⟨cost, finalEnv, finalHeap, bounded, tail, finalInv⟩ := ih (start + 1) (by omega) rowEnv rowHeap rowInv
      refine ⟨(8 + (rowCost + 1) + 1) + cost + 1, finalEnv, finalHeap, ?_, ?_, finalInv⟩
      · rw [Nat.mul_succ]
        have rowMaximum := Nat.mul_le_mul_left (42 * n + 51) (Nat.sub_le (hi + 1) start)
        omega
      · apply BlockContinuesIn.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, outerBody, Int.natCast_add] using row.append tail

/-- The heap backend executes the reference's linked helper and every allocation.
The bound is in decimal digit count; collection primitive internals are atomic. -/
theorem source_bound (n : Nat) (positive : 0 < n) :
    ∃ cost best finalHeap, FinishesIn (HeapCalls.step P4Ast.module 0)
      ⟨.operation (.call "naive") [.scalar (.int n)], [], []⟩ cost
      (.ok ⟨.ok (.scalar (.int best)), [], finalHeap, 0⟩) ∧
      cost ≤ 256 * (10 ^ (2 * n) * (n + 1)) := by
  let lo : Nat := 10 ^ (n - 1)
  let hi : Nat := 10 ^ n - 1
  let width : Nat := hi + 1 - lo
  have powerPositive : 1 ≤ (10 : Nat) ^ n := Nat.one_le_pow _ _ (by decide)
  have high : hi + 1 = 10 ^ n := by dsimp [hi]; omega
  have lowerBound : lo ≤ hi + 1 := by
    rw [high]
    exact Nat.pow_le_pow_right (by decide) (Nat.sub_le n 1)
  let source := executableBody ((findFunctionIn program.body "naive").getD ([], [])).2
  let env := functionEnv ((findFunctionIn program.body "naive").getD ([], [])).1
    ((findFunctionIn program.body "naive").getD ([], [])).2 [.scalar (.int n)] []
  let envLo := setLocal env "lo" (.scalar (.int lo))
  let envHi := setLocal envLo "hi" (.scalar (.int hi))
  let envMax := setLocal envHi "max_" (.scalar (.int 0))
  have lower : EvaluatesIn program env
      (.ifExp (.compare (.name "n") [(.eq, .constant (.int 0))]) (.constant (.int 0))
        (.binOp (.constant (.int 10)) .pow (.binOp (.name "n") .sub (.constant (.int 1)))))
      [] 16 (.scalar (.int lo)) [] := by
    apply EvaluatesIn.conditional (ct := 5) (cb := 9) (flag := false) (value := .scalar (.bool false)) (middle := [])
    · exact EvaluatesIn.compare (EvaluatesIn.lookup rfl) EvaluatesIn.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply EvaluatesIn.binary (ca := 1) (cb := 5) (bv := .scalar (.int ((n - 1 : Nat) : Int)))
        (middle := []) (afterRight := []) EvaluatesIn.constant
      · exact EvaluatesIn.binary (EvaluatesIn.lookup rfl) EvaluatesIn.constant
          (by simp [Mutable.binary, scalarBinary]; omega)
      · simp [Mutable.binary, scalarBinary, lo, show ¬ ((n - 1 : Nat) : Int) < 0 by omega]
  have upper : EvaluatesIn program envLo
      (.binOp (.binOp (.constant (.int 10)) .pow (.name "n")) .sub (.constant (.int 1)))
      [] 9 (.scalar (.int hi)) [] := by
    apply EvaluatesIn.binary (ca := 5) (cb := 1) (av := .scalar (.int ((10 ^ n : Nat) : Int)))
      (middle := []) (afterRight := [])
    · exact EvaluatesIn.binary EvaluatesIn.constant (EvaluatesIn.lookup rfl) (by simp [Mutable.binary, scalarBinary, show ¬ (n : Int) < 0 by omega])
    · exact EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, hi, Int.natCast_sub powerPositive]
  obtain ⟨loopCost, finalEnv, finalHeap, loopBound, loops, finalInv⟩ :=
    outer_loop n width lo hi (by dsimp [width]; omega) (by omega) envMax [] ⟨rfl, 0, rfl⟩
  obtain ⟨best, getBest⟩ := finalInv.best
  have range : BlockReturnsIn program envMax (source.drop 3) [] (8 + ((loopCost + 2) + 1) + 1)
      (.scalar (.int best)) finalHeap := by
    apply BlockReturnsIn.for_range (middle := [])
      (values := [.scalar (.int lo), .scalar (.int ((hi + 1 : Nat) : Int))])
    · exact .cons (EvaluatesIn.lookup rfl) (.cons
        (EvaluatesIn.binary (EvaluatesIn.lookup rfl) EvaluatesIn.constant (by simp [Mutable.binary, scalarBinary])) .nil)
    · rfl
    · exact loops.returns (BlockReturnsIn.return_value (EvaluatesIn.lookup getBest))
  have block := BlockReturnsIn.assign lower (BlockReturnsIn.assign upper
    (BlockReturnsIn.assign (EvaluatesIn.constant (value := .int 0)) range))
  have called := BlockReturnsIn.function (module := program) (name := "naive") (values := [.scalar (.int n)]) rfl rfl block
  refine ⟨_, best, finalHeap, (called 0 []).finishes (.done rfl), ?_⟩
  have widthBound : width ≤ 10 ^ n := by
    exact (Nat.sub_le (hi + 1) lo).trans (by rw [high])
  have enlarged := Nat.mul_le_mul_left ((42 * n + 51) * (hi + 1) + 12) widthBound
  rw [high] at enlarged
  have square : 10 ^ (2 * n) = (10 ^ n) * 10 ^ n := by rw [two_mul, pow_add]
  rw [square]
  have powSquare : 10 ^ n ≤ (10 ^ n) * 10 ^ n := by nlinarith
  have positiveSquare : 1 ≤ (10 ^ n) * 10 ^ n := by nlinarith
  rw [high] at loopBound
  nlinarith only [loopBound, enlarged, powSquare, positiveSquare, Nat.zero_le (n * (10 ^ n * 10 ^ n))]

end Reference
/-- Calls execution: scalar arithmetic and string primitives are atomic. -/
def optimized (n : Nat) : Execution Constant := Calls.function Solvers.P4Ast.module "solve" [.int n]

/-- HeapCalls execution: list primitives are atomic; digits_le is source-executed. -/
def reference (n : Nat) : Execution Mutable.Completion :=
  (HeapCalls.call P4Ast.module "naive" [.scalar (.int n)] []).execution

/-- A source envelope, not a strict comparison with the different reference backend. -/
theorem optimized_bigO : Execution.IsBigO optimized (fun _ => True) id (fun n => 10 ^ (2 * n) * (n + 1)) := by
  refine ⟨150, 1, fun n _ large => ?_⟩
  obtain ⟨cost, value, run, bound⟩ := Optimized.source_bound n (by simpa using large)
  refine ⟨cost, .ok value, run, Nat.le_trans bound ?_⟩
  simp only [id_eq]
  exact Nat.mul_le_mul_left 150 (Nat.le_mul_of_pos_right _ (by omega))

/-- The same digit-count envelope, proved independently in HeapCalls. -/
theorem reference_bigO : Execution.IsBigO reference (fun _ => True) id (fun n => 10 ^ (2 * n) * (n + 1)) := by
  refine ⟨256, 1, fun n _ large => ?_⟩
  obtain ⟨cost, best, heap, run, bound⟩ := Reference.source_bound n (by simpa using large)
  exact ⟨cost, .ok ⟨.ok (.scalar (.int best)), [], heap, 0⟩, run, bound⟩

end solvers.«4».complexity
