import «.formalsw».solvers.«4.py.ast»
import lang.py.FormalSW.CallsRuntimeCostStrings
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Lift

namespace solvers.«4».runtimeCost
open FormalSW.PyAst FormalSW.PyAst.Calls Runtime.Scaled
set_option maxRecDepth 4096
set_option maxHeartbeats 1200000

private def unit (n : Nat) := 16*(n+1)
private theorem unit_large (n : Nat) : 16 ≤ unit n := by dsimp [unit]; omega
private theorem unit_positive (n : Nat) : 1 ≤ unit n := by have := unit_large n; omega
private theorem compare_price (n : Nat) (a b : Int) :
    PrimitiveWork.comparison .word (.int a) (.int b)+1 ≤ unit n := by
  change 3 ≤ unit n
  have := unit_large n
  omega
private theorem binary_price (n : Nat) (a b : Int) (op : BinOp)
    (ordinary : op = .mult ∨ op = .sub) :
    ∃ work, PrimitiveWork.binary .word op (.int a) (.int b) = some work ∧ work+1 ≤ unit n := by
  rcases ordinary with rfl | rfl <;> exact ⟨1, rfl, by have := unit_large n; omega⟩

private theorem text_bounds (n : Nat) (positive : 0 < n) (value : Int)
    (nonnegative : 0 ≤ value) (bounded : value < (10^(2*n) : Nat)) :
    PrimitiveWork.bitLength value ≤ 8*n+1 ∧ (toString value).length ≤ 2*n := by
  lift value to Nat using nonnegative
  have bound : value < 10^(2*n) := by exact_mod_cast bounded
  have w := PrimitiveWork.bitLength_decimal value (2*n) bound
  exact ⟨by omega, PrimitiveWork.decimal_length value (2*n) (by omega) bound⟩

private theorem product_bounds (n : Nat) (a b hi : Int) (ha : 0 ≤ a ∧ a ≤ hi)
    (hb : 0 ≤ b ∧ b ≤ a) (top : hi < (10^n : Nat)) :
    0 ≤ a*b ∧ a*b < (10^(2*n) : Nat) := by
  have hn : 0 ≤ hi := le_trans ha.1 ha.2
  have positive : (0 : Int) < (10^n : Nat) := by omega
  refine ⟨mul_nonneg ha.1 hb.1, ?_⟩
  have bound : a*b < ((10^n : Nat) : Int)*((10^n : Nat) : Int) := calc
    a*b ≤ hi*hi := mul_le_mul ha.2 (le_trans hb.2 ha.2) hb.1 hn
    _ ≤ hi*((10^n : Nat) : Int) := mul_le_mul_of_nonneg_left (le_of_lt top) hn
    _ < ((10^n : Nat) : Int)*((10^n : Nat) : Int) := mul_lt_mul_of_pos_right top positive
  simpa [two_mul, pow_add] using bound

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

private theorem str_cost (n : Nat) (env : ScalarEnv) (value : Int) (found : env "prod" = some (.int value))
    (width : PrimitiveWork.bitLength value ≤ 8*n+1) :
    EvaluatesIn .word program (unit n) env (.builtinCall .str [.name "prod"]) 4 (.string (toString value)) := by
  apply EvaluatesIn.builtin_single (unit_positive n) (EvaluatesIn.lookup (unit_positive n) found) rfl
  refine ⟨_, rfl, ?_⟩
  change PrimitiveWork.bitLength value+2 ≤ unit n
  dsimp [unit]
  omega

private theorem palindrome_cost (n : Nat) (env : ScalarEnv) (s : String) (found : env "s" = some (.string s))
    (length : s.length ≤ 2*n) :
    EvaluatesIn .word program (unit n) env palindromeTest 16 (.bool (decide (s = String.ofList s.toList.reverse))) := by
  apply EvaluatesIn.compare (unit_positive n) (ca := 1) (cb := 12)
    (bv := .string (String.ofList s.toList.reverse)) (EvaluatesIn.lookup (unit_positive n) found)
  · exact ⟨s.length+27, by dsimp [unit]; omega, Runtime.EvaluatesIn.reverse_string found⟩
  · simp [scalarCompare]
  · simp [PrimitiveWork.comparison, PrimitiveWork.scalarSize]
    dsimp [unit]
    omega

private theorem inner_loop (n : Nat) (positive : 0 < n) (remaining : Nat) (b a lo hi : Int)
    (endpoint : b = lo-1 + remaining) (low : 0 ≤ lo) (ab : 0 ≤ a ∧ a ≤ hi)
    (bb : b ≤ a) (top : hi < (10^n : Nat))
    (env : ScalarEnv) (following : List Stmt) (budget : Nat)
    (bounds : Bounds env lo hi) (getA : env "a" = some (.int a))
    (after : ∀ nextEnv, Bounds nextEnv lo hi → BlockWithin .word program (unit n) nextEnv following budget) :
    BlockWithin .word program (unit n) env (.rangeStepLoop "b" b (lo-1) (-1) innerBody [] :: following)
      (44 * remaining + 1 + budget) := by
  induction remaining generalizing b env with
  | zero =>
      simpa [Nat.add_comm] using BlockWithin.range_done (name := "b") (next := b) (stop := lo-1) (stride := -1)
        (body := innerBody) (otherwise := []) (by simp [rangeContinues]; omega) (by have := unit_large n; change 4 ≤ unit n; omega) (after env bounds)
  | succ remaining ih =>
      obtain ⟨best, getBest⟩ := bounds.best
      have product := product_bounds n a b hi ab ⟨by omega, bb⟩ top
      refine (BlockWithin.range_step (bound := 7 + (44 * remaining + 36 + budget)) (by simp [rangeContinues]; omega) (by have := unit_large n; change 4 ≤ unit n; omega) ?_).mono ?_
      · let envB := env.set "b" (.int b)
        have boundsB : Bounds envB lo hi := bounds.setAux "b" (.int b) (by decide) (by decide) (by decide)
        refine BlockWithin.assign (unit_positive n) (budget := 5) (bound := 44 * remaining + 36 + budget) (value := .int (a * b)) ?_ ?_
        · exact EvaluatesIn.binary (unit_positive n) (EvaluatesIn.lookup (unit_positive n) (by simpa [envB, ScalarEnv.set] using getA))
            (EvaluatesIn.lookup (unit_positive n) rfl) rfl
            (binary_price n a b .mult (by simp))
        · let envP := envB.set "prod" (.int (a * b))
          have boundsP : Bounds envP lo hi := boundsB.setAux "prod" (.int (a * b)) (by decide) (by decide) (by decide)
          have guard : EvaluatesIn .word program (unit n) envP (.compare (.name "prod") [(.ltE, .name "best")]) 5
              (.bool (decide (a * b ≤ best))) :=
            EvaluatesIn.compare (unit_positive n) (EvaluatesIn.lookup (unit_positive n) rfl)
              (EvaluatesIn.lookup (unit_positive n) (by simpa [envP, envB, ScalarEnv.set] using getBest))
              (by simp [scalarCompare, evalCmpOp]) (compare_price n _ _)
          by_cases beaten : a * b ≤ best
          · refine (BlockWithin.branch (unit_positive n) (bound := budget + 1) (by simpa [beaten] using guard) ?_).mono (large := 44 * remaining + 36 + budget) ?_
            · exact BlockWithin.break_loop (unit_positive n) rfl (after envP boundsP)
            · omega
          · refine (BlockWithin.branch (unit_positive n) (bound := 6 + (16 + 2 + (44 * remaining + 5 + budget)))
              (by simpa [beaten] using guard) ?_).mono (large := 44 * remaining + 36 + budget) ?_
            · refine BlockWithin.assign (unit_positive n) (bound := 16 + 2 + (44 * remaining + 5 + budget)) (str_cost n envP (a * b) rfl (text_bounds n positive (a*b) product.1 product.2).1) ?_
              let envS := envP.set "s" (.string (toString (a * b)))
              have boundsS : Bounds envS lo hi := boundsP.setAux "s" (.string (toString (a * b)))
                (by decide) (by decide) (by decide)
              refine BlockWithin.branch (unit_positive n) (bound := 44 * remaining + 5 + budget) (palindrome_cost n envS (toString (a * b)) rfl (text_bounds n positive (a*b) product.1 product.2).2) ?_
              by_cases palindrome : toString (a * b) = String.ofList (toString (a * b)).toList.reverse
              · have test : decide (toString (a * b) = String.ofList (toString (a * b)).toList.reverse) = true := decide_eq_true palindrome
                simp only [test, ↓reduceIte]
                refine (BlockWithin.assign (unit_positive n) (EvaluatesIn.lookup (unit_positive n) rfl)
                  (BlockWithin.break_loop (unit_positive n) rfl (after _ (boundsS.setBest (a * b))))).mono (large := 44 * remaining + 5 + budget) ?_
                omega
              · have test : decide (toString (a * b) = String.ofList (toString (a * b)).toList.reverse) = false := decide_eq_false palindrome
                simp only [test, Bool.false_eq_true, ↓reduceIte]
                have next := ih (b - 1) (by omega) (by omega) envS boundsS
                  (by simpa [envS, envP, envB, ScalarEnv.set] using getA)
                have resumed := BlockWithin.resume (unit_positive n) next
                apply BlockWithin.mono (small := (44 * remaining + 1 + budget) + 1)
                · simpa only [show b + (-1) = b - 1 by omega] using resumed
                · omega
            · omega
      · omega

private theorem finish (n : Nat) (bounds : Bounds env lo hi) :
    BlockWithin .word program (unit n) env [.return (.name "best")] 2 := by
  obtain ⟨best, found⟩ := bounds.best
  exact BlockWithin.return_value (unit_positive n) (EvaluatesIn.lookup (unit_positive n) found)

private theorem outer_loop (n : Nat) (positive : 0 < n) (remaining width : Nat) (a lo hi : Int)
    (endpoint : a = lo - 1 + remaining) (small : remaining ≤ width)
    (extent : hi = lo-1+width) (low : 0 ≤ lo) (top : hi < (10^n : Nat))
    (env : ScalarEnv) (bounds : Bounds env lo hi) :
    BlockWithin .word program (unit n) env [.rangeStepLoop "a" a (lo - 1) (-1) outerBody [], .return (.name "best")]
      ((44 * width + 30) * remaining + 3) := by
  induction remaining generalizing a env with
  | zero =>
      simpa using BlockWithin.range_done (by simp [rangeContinues]; omega) (by have := unit_large n; change 4 ≤ unit n; omega) (finish n bounds)
  | succ remaining ih =>
      obtain ⟨best, getBest⟩ := bounds.best
      refine (BlockWithin.range_step (bound := 44 * width + 29 + ((44 * width + 30) * remaining + 3)) (by simp [rangeContinues]; omega) (by have := unit_large n; change 4 ≤ unit n; omega) ?_).mono ?_
      · let envA := env.set "a" (.int a)
        have boundsA : Bounds envA lo hi := bounds.setAux "a" (.int a) (by decide) (by decide) (by decide)
        have guard : EvaluatesIn .word program (unit n) envA
            (.compare (.binOp (.name "a") .mult (.name "hi")) [(.lt, .name "best")]) 9
            (.bool (decide (a * hi < best))) :=
          EvaluatesIn.compare (unit_positive n) (EvaluatesIn.binary (unit_positive n) (EvaluatesIn.lookup (unit_positive n) rfl) (EvaluatesIn.lookup (unit_positive n) boundsA.upper) rfl (binary_price n a hi .mult (by simp)))
            (EvaluatesIn.lookup (unit_positive n) (by simpa [envA, ScalarEnv.set] using getBest))
            (by simp [scalarCompare, evalCmpOp]) (compare_price n _ _)
        by_cases beaten : a * hi < best
        · refine (BlockWithin.branch (unit_positive n) (by simpa [beaten] using guard)
            (BlockWithin.break_loop (unit_positive n) rfl (finish n boundsA))).mono
            (large := 44 * width + 29 + ((44 * width + 30) * remaining + 3)) ?_
          omega
        · refine (BlockWithin.branch (unit_positive n) (bound := 12 + 2 + (44 * (remaining + 1) + 1 + (((44 * width + 30) * remaining + 3) + 1)))
            (by simpa [beaten] using guard) ?_).mono
            (large := 44 * width + 29 + ((44 * width + 30) * remaining + 3)) ?_
          · refine BlockWithin.for_range (unit_positive n) (values := [.int a, .int (lo - 1), .int (-1)]) ?_ (by have := unit_large n; change 3 ≤ unit n; omega) (by have := unit_large n; change 4 ≤ unit n; omega) rfl ?_
            · exact .cons (EvaluatesIn.lookup (unit_positive n) rfl) (.cons
                (EvaluatesIn.binary (unit_positive n) (EvaluatesIn.lookup (unit_positive n) boundsA.lower) (EvaluatesIn.constant (unit_positive n)) rfl (binary_price n lo 1 .sub (by simp)))
                (.cons (EvaluatesIn.unary (unit_positive n) (EvaluatesIn.constant (unit_positive n)) rfl (work := 2) (by intros; rfl) (by have := unit_large n; omega)) .nil))
            · apply inner_loop n positive (remaining + 1) a a lo hi endpoint low ⟨by omega, by omega⟩ (by omega) top envA _
                (((44 * width + 30) * remaining + 3) + 1) boundsA rfl
              intro nextEnv nextBounds
              have next := ih (a - 1) (by omega) (by omega) nextEnv nextBounds
              simpa only [show a + (-1) = a - 1 by omega] using BlockWithin.resume (unit_positive n) next
          · have grow := Nat.mul_le_mul_left 44 small
            omega
      · rw [Nat.mul_succ]
        have grow := Nat.mul_le_mul_left 44 small
        omega


private theorem pow_price (n k : Nat) (bound : k ≤ n) :
    ∃ work, PrimitiveWork.binary .word .pow (.int 10) (.int k) = some work ∧ work+1 ≤ unit n := by
  refine ⟨(k+1).log2+1, ?_, ?_⟩
  · simp [PrimitiveWork.binary, PrimitiveWork.integerBinary, scalarNumber]
  · have := Nat.log2_le_self (k+1)
    dsimp [unit]
    omega

private theorem largest_cost (n : Nat) (positive : 0 < n) (lo hi : Int) (width : Nat)
    (endpoint : hi = lo-1+width) (low : 0 ≤ lo) (top : hi < (10^n : Nat)) :
    ∃ value, ∀ stack, Reaches .word program (unit n)
      ⟨.invoke "largest_palindrome_product" [.int lo, .int hi], stack⟩
      ((44*width+30)*width+21) ⟨.value value, stack⟩ := by
  let env := argumentEnv [⟨"lo", some (.name "int"), none⟩, ⟨"hi", some (.name "int"), none⟩] [.int lo, .int hi]
  have loops := outer_loop n positive width width hi lo hi endpoint (Nat.le_refl _)
    endpoint low top (env.set "best" (.int 0)) ⟨rfl, rfl, 0, rfl⟩
  have range := BlockWithin.for_range (unit_positive n) (name := "a")
    (expressions := [.name "hi", .binOp (.name "lo") .sub (.constant (.int 1)),
      .unaryOp .uSub (.constant (.int 1))])
    (.cons (EvaluatesIn.lookup (unit_positive n) (value := .int hi) rfl)
      (.cons (EvaluatesIn.binary (unit_positive n)
        (EvaluatesIn.lookup (unit_positive n) (value := .int lo) rfl)
        (EvaluatesIn.constant (unit_positive n)) rfl (binary_price n lo 1 .sub (by simp)))
        (.cons (EvaluatesIn.unary (unit_positive n) (EvaluatesIn.constant (unit_positive n)) rfl
          (work := 2) (by intros; rfl) (by have := unit_large n; omega)) .nil)))
    (by have := unit_large n; change 3 ≤ unit n; omega)
    (by have := unit_large n; change 4 ≤ unit n; omega) rfl loops
  have block := BlockWithin.assign (unit_positive n) (name := "best")
    (EvaluatesIn.constant (unit_positive n) (env := env)) range
  obtain ⟨cost, value, bounded, run⟩ := block
  refine ⟨value, fun stack => ⟨3+cost, ?_, ?_⟩⟩
  · have h := unit_large n
    simp only [Nat.mul_add] at bounded ⊢
    omega
  · apply RuntimeCost.Reaches.next (following := ⟨.block (.assign [.name "best"] (.constant (.int 0)) ::
        [.forIn (.name "a") (.call (.name "range") [.name "hi", .binOp (.name "lo") .sub (.constant (.int 1)),
          .unaryOp .uSub (.constant (.int 1))]) outerBody [], .return (.name "best")]) env, stack⟩)
    · rfl
    · rfl
    · exact run stack

/-- The complete optimized source, charging decimal conversion, string reversal,
comparison, range setup and exponentiation under fixed arithmetic. -/
theorem source_bound (n : Nat) (positive : 0 < n) :
    Runtime.TimeBound .word Solvers.P4Ast.module "solve" [.int n]
      (unit n * (150*10^(2*n))) := by
  let lo : Nat := 10^(n-1)
  let hi : Nat := 10^n-1
  have powerPositive : 1 ≤ (10 : Nat)^n := Nat.one_le_pow _ _ (by decide)
  have lowerBound : lo ≤ hi+1 := by
    dsimp [lo, hi]
    rw [Nat.sub_add_cancel powerPositive]
    exact Nat.pow_le_pow_right (by decide) (Nat.sub_le n 1)
  let width : Nat := hi+1-lo
  have endpoint : (hi : Int) = (lo : Int)-1+width := by omega
  have topNat : hi < 10^n := by dsimp [hi]; omega
  have top : (hi : Int) < (10^n : Nat) := by exact_mod_cast topNat
  obtain ⟨value, helper⟩ := largest_cost n positive lo hi width endpoint (Int.natCast_nonneg lo) top
  let env := argumentEnv [⟨"n", some (.name "int"), none⟩] [.int n]
  let envLo := env.set "lo" (.int lo)
  let envHi := envLo.set "hi" (.int hi)
  let source := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
  have lower : EvaluatesIn .word program (unit n) env
      (.ifExp (.compare (.name "n") [(.eq, .constant (.int 0))]) (.constant (.int 0))
        (.binOp (.constant (.int 10)) .pow (.binOp (.name "n") .sub (.constant (.int 1))))) 16 (.int lo) := by
    apply EvaluatesIn.conditional (unit_positive n) (ct := 5) (cb := 9) (flag := false)
    · exact EvaluatesIn.compare (unit_positive n) (EvaluatesIn.lookup (unit_positive n) rfl)
        (EvaluatesIn.constant (unit_positive n))
        (by simp [scalarCompare, evalCmpOp]; omega) (compare_price n _ _)
    · apply EvaluatesIn.binary (unit_positive n) (ca := 1) (cb := 5) (bv := .int ((n-1 : Nat) : Int))
        (EvaluatesIn.constant (unit_positive n))
      · exact EvaluatesIn.binary (unit_positive n) (EvaluatesIn.lookup (unit_positive n) rfl)
          (EvaluatesIn.constant (unit_positive n)) (by simp [scalarBinary]; omega)
          (binary_price n n 1 .sub (by simp))
      · simp [scalarBinary, lo]
      · exact pow_price n (n-1) (by omega)
  have upper : EvaluatesIn .word program (unit n) envLo
      (.binOp (.binOp (.constant (.int 10)) .pow (.name "n")) .sub (.constant (.int 1))) 9 (.int hi) := by
    apply EvaluatesIn.binary (unit_positive n) (ca := 5) (cb := 1) (av := .int ((10^n : Nat) : Int))
    · exact EvaluatesIn.binary (unit_positive n) (EvaluatesIn.constant (unit_positive n))
        (EvaluatesIn.lookup (unit_positive n) rfl) (by simp [scalarBinary]) (pow_price n n le_rfl)
    · exact EvaluatesIn.constant (unit_positive n)
    · simp [scalarBinary, hi, Int.natCast_sub powerPositive]
    · exact binary_price n _ _ .sub (by simp)
  have start : Reaches .word program (unit n) ⟨.invoke "solve" [.int n], []⟩ 1 ⟨.block source env, []⟩ := by
    refine Reaches.next (budget := 0) (following := ⟨.block source env, []⟩) rfl ?_ .refl
    exact ⟨2, rfl, by have := unit_large n; omega⟩
  have tail : Reaches .word program (unit n) ⟨.block (source.drop 2) envHi, []⟩ 6
      ⟨.invoke "largest_palindrome_product" [.int lo, .int hi], []⟩ := by
    refine ⟨7, by have := unit_large n; omega, ?_⟩
    repeat first | exact RuntimeCost.Reaches.refl | runtime_step
  have lead : Reaches .word program (unit n) ⟨.invoke "solve" [.int n], []⟩ 36
      ⟨.invoke "largest_palindrome_product" [.int lo, .int hi], []⟩ :=
    ((start.trans (Runtime.Scaled.assign (unit_positive n) lower "lo" (source.drop 1) [])).trans
      (Runtime.Scaled.assign (unit_positive n) upper "hi" (source.drop 2) [])).trans tail
  have widthBound : width ≤ 10^n := by
    dsimp [width]
    exact (Nat.sub_le (hi+1) lo).trans (by dsimp [hi]; rw [Nat.sub_add_cancel powerPositive])
  have square : 10^(2*n) = (10^n)*10^n := by rw [two_mul, pow_add]
  have steps : 36+((44*width+30)*width+21)+1 ≤ 150*10^(2*n) := by
    rw [square]
    nlinarith only [powerPositive, widthBound, Nat.mul_self_le_mul_self widthBound]
  obtain ⟨cost, bound, run⟩ := lead.trans (helper [])
  refine ⟨cost+1, .ok value, run.finishes (.done rfl rfl), ?_⟩
  calc
    cost+1 ≤ unit n*(36+((44*width+30)*width+21))+unit n := Nat.add_le_add bound (unit_positive n)
    _ = unit n*(36+((44*width+30)*width+21)+1) := by simp only [Nat.mul_add, Nat.mul_one]
    _ ≤ _ := Nat.mul_le_mul_left (unit n) steps

end solvers.«4».runtimeCost


theorem solvers.«4».solve.«BigO(10 ^ (2 * n) * (n + 1))» :
    FormalSW.PyAst.Calls.Runtime.IsBigO .word Solvers.P4Ast.module "solve"
      (fun n : Nat => [.int n])
      (fun n => FormalSW.PyAst.Calls.Condition Solvers.P4Ast.module
        (fun name => if name = "n" then some (.int n) else none) Solvers.P4Ast.requires_solve_0)
      id (fun n => 10 ^ (2 * n) * (n + 1)) := by
  refine ⟨2400, 1, fun n _ large => ?_⟩
  apply (solvers.«4».runtimeCost.source_bound n (by simp only [id_eq] at large; omega)).mono
  dsimp [solvers.«4».runtimeCost.unit]
  nlinarith only []
