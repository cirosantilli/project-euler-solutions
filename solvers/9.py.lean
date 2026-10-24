import «.formalsw».solvers.«9.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p9.py»
import ProjectEulerSolutions.Equiv.P9
import FormalSW
import Mathlib.Tactic

namespace solvers.«9»
open FormalSW.PyAst
private abbrev program := Solvers.P9Ast.module
private def scalarEnv (total : Int) : ScalarEnv :=
  fun name => if name = "total" then some (.int total) else none
private def heapEnv (total : Int) : Mutable.Env :=
  fun name => if name = "total" then some (.scalar (.int total)) else none
private abbrev nonnegative (total : Int) :=
  Calls.Condition program (scalarEnv total) Solvers.P9Ast.requires_solve_0
private abbrev hasTriplet (total : Int) :=
  HeapCalls.Condition program (heapEnv total) Solvers.P9Ast.requires_solve_1 []

private theorem extends_naive : HeapCalls.Extends program P9Ast.module := by
  apply HeapCalls.Extends.of_classless (classless := rfl)
  change Calls.Extends { body := program.body.take 4 ++ P9Ast.module.body } P9Ast.module
  apply Calls.Extends.prepend
  · intro name member
    change name ∈ ["solve", "_formalsw_requires_target"] at member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> decide
  · intro name member
    change name ∈ ["annotations"] at member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    subst name
    decide

theorem nonnegative_iff (total : Int) : nonnegative total ↔ 0 ≤ total := by
  have evaluated : Calls.Evaluates program (scalarEnv total) Solvers.P9Ast.requires_solve_0
      (.bool (decide (0 ≤ total))) :=
    Calls.Evaluates.compare (Calls.Evaluates.lookup rfl) Calls.Evaluates.constant rfl
  simpa using evaluated.condition_iff

theorem triplet_condition_evaluates (total : Nat) : ∃ heap,
    HeapCalls.Evaluates program (heapEnv total) Solvers.P9Ast.requires_solve_1 []
      (.scalar (.bool (decide (0 < ProjectEulerStatements.P9.naive total)))) heap := by
  have naive := (data.«project-euler-statements».data.py.project_euler_statements.p9.naive.naturals total []).extend extends_naive
  have wrapped : HeapCalls.ReturnsValue program "_formalsw_requires_target" [.scalar (.int total)] []
      (.scalar (.int (ProjectEulerStatements.P9.naive total)))
      [.set ((data.«project-euler-statements».data.py.project_euler_statements.p9.products total).map Constant.int)] := by
    apply HeapCalls.BlockReturns.function rfl rfl
    exact HeapCalls.BlockReturns.return_value
      (HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup rfl) .nil) naive)
  refine ⟨[.set ((data.«project-euler-statements».data.py.project_euler_statements.p9.products total).map Constant.int)], HeapCalls.Evaluates.compare
    (HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup rfl) .nil) wrapped)
    HeapCalls.Evaluates.constant ?_⟩
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem hasTriplet_iff (total : Nat) : hasTriplet total ↔ 0 < ProjectEulerStatements.P9.naive total := by
  obtain ⟨heap, evaluated⟩ := triplet_condition_evaluates total
  simpa using evaluated.condition_iff

theorem requires_solve_1_supported (total : Int) (positive : nonnegative total) :
    (HeapCalls.evaluate program (heapEnv total) Solvers.P9Ast.requires_solve_1 []).Supported := by
  have nonneg := (nonnegative_iff total).mp positive
  obtain ⟨heap, evaluated⟩ := triplet_condition_evaluates total.toNat
  have returned : (HeapCalls.evaluate program (heapEnv total.toNat) Solvers.P9Ast.requires_solve_1 []).Returns :=
    ⟨_, [], heap, 0, evaluated.finishes⟩
  simpa [Int.toNat_of_nonneg nonneg] using returned.supported

/- The Python loop sorts Euclid's two legs before checking the triangle.
These facts connect that order to the candidate used in the existing proof. -/
private theorem unequal_legs (a c : Nat) (positive : 0 < a)
    (squares : a * a + a * a = c * c) : False := by
  induction a using Nat.strong_induction_on generalizing c with
  | h a ih =>
      have ceven : 2 ∣ c ^ 2 := ⟨a * a, by nlinarith [squares]⟩
      obtain ⟨d, cd⟩ := Nat.prime_two.dvd_of_dvd_pow ceven
      have aeven : 2 ∣ a ^ 2 := ⟨d * d, by nlinarith [squares]⟩
      obtain ⟨b, ab⟩ := Nat.prime_two.dvd_of_dvd_pow aeven
      exact ih b (by omega) d (by omega) (by nlinarith [squares])

open ProjectEulerSolutions.P9

theorem euclid_cast (total m n : Nat) (before : n < m) :
    (euclidA total m n : Int) = (euclidK total m n : Int) * ((m : Int) * m - (n : Int) * n) ∧
    (euclidB total m n : Int) = (euclidK total m n : Int) * (2 * (m : Int) * n) ∧
    (euclidC total m n : Int) = (euclidK total m n : Int) * ((m : Int) * m + (n : Int) * n) := by
  have square : n * n ≤ m * m := Nat.mul_self_le_mul_self (Nat.le_of_lt before)
  simp [euclidA, euclidB, euclidC, Nat.cast_sub square]

theorem euclid_squares (total m n : Nat) (before : n < m) :
    euclidA total m n * euclidA total m n + euclidB total m n * euclidB total m n =
      euclidC total m n * euclidC total m n := by
  have casts := euclid_cast total m n before
  have identity : (euclidA total m n : Int) * euclidA total m n +
      (euclidB total m n : Int) * euclidB total m n =
      (euclidC total m n : Int) * euclidC total m n := by
    rw [casts.1, casts.2.1, casts.2.2]
    ring
  exact_mod_cast identity

theorem euclid_order (total m n : Nat) (positive : 0 < n) (before : n < m)
    (scale : 0 < euclidK total m n) :
    0 < euclidA total m n ∧ 0 < euclidB total m n ∧
      ((euclidA total m n < euclidB total m n ∧ euclidB total m n < euclidC total m n) ∨
       (euclidB total m n < euclidA total m n ∧ euclidA total m n < euclidC total m n)) := by
  have apositive : 0 < euclidA total m n := by
    apply Nat.mul_pos scale
    apply Nat.sub_pos_of_lt
    nlinarith
  have bpositive : 0 < euclidB total m n := by
    have mpositive : 0 < m := lt_trans positive before
    unfold euclidB
    exact Nat.mul_pos scale (by positivity)
  have squares := euclid_squares total m n before
  have ac : euclidA total m n < euclidC total m n := by nlinarith
  have bc : euclidB total m n < euclidC total m n := by nlinarith
  have different : euclidA total m n ≠ euclidB total m n := by
    intro equal
    rw [← equal] at squares
    exact unequal_legs _ _ apositive squares
  exact ⟨apositive, bpositive, (lt_or_gt_of_ne different).elim
    (fun ab => .inl ⟨ab, bc⟩) (fun ba => .inr ⟨ba, ac⟩)⟩

theorem euclid_sorted (total m n : Nat) (positive : 0 < n) (before : n < m) :
    ScalarSort.sort [.int (euclidA total m n), .int (euclidB total m n), .int (euclidC total m n)] =
      .ok [.int (min (euclidA total m n) (euclidB total m n)),
        .int (max (euclidA total m n) (euclidB total m n)), .int (euclidC total m n)] := by
  by_cases zero : euclidK total m n = 0
  · simp [euclidA, euclidB, euclidC, zero, ScalarSort.sort]
    rfl
  · obtain ⟨_, _, order⟩ := euclid_order total m n positive before (Nat.pos_of_ne_zero zero)
    have ac : euclidA total m n ≤ euclidC total m n := by rcases order with h | h <;> omega
    have bc : euclidB total m n ≤ euclidC total m n := by rcases order with h | h <;> omega
    simpa only [Nat.cast_min, Nat.cast_max] using ScalarSort.three_top
      (euclidA total m n) (euclidB total m n) (euclidC total m n)
      (by exact_mod_cast ac) (by exact_mod_cast bc)

theorem euclid_candidate (total m n : Nat) (totalPositive : 0 < total)
    (positive : 0 < n) (before : n < m) (divides : total % euclidS m n = 0) :
    candidate total m n = euclidA total m n * euclidB total m n * euclidC total m n ∧
      min (euclidA total m n) (euclidB total m n) > 0 ∧
      min (euclidA total m n) (euclidB total m n) +
        max (euclidA total m n) (euclidB total m n) + euclidC total m n = total := by
  have divisor : euclidS m n ∣ total := Nat.dvd_of_mod_eq_zero divides
  have mpositive : 0 < m := lt_trans positive before
  have spos : 0 < euclidS m n := by unfold euclidS; positivity
  have scale : 0 < euclidK total m n := Nat.div_pos (Nat.le_of_dvd totalPositive divisor) spos
  obtain ⟨apos, bpos, order⟩ := euclid_order total m n positive before scale
  have sum : euclidA total m n + euclidB total m n + euclidC total m n = total := by
    have identity := euclid_sum_identity (euclidK total m n) m n (Nat.le_of_lt before)
    change euclidK total m n * (m * m - n * n) + euclidK total m n * (2 * m * n) +
      euclidK total m n * (m * m + n * n) = total
    simpa only [pow_two, euclidK, Nat.div_mul_cancel divisor] using identity
  have valid : candidateValid total m n :=
    ⟨divides, apos, order, euclid_squares total m n before, sum⟩
  refine ⟨by simp [candidate, valid], lt_min apos bpos, ?_⟩
  simpa only [min_add_max] using sum

private def triple (a b c : Int) : List Mutable.Value :=
  [.scalar (.int a), .scalar (.int b), .scalar (.int c)]
private abbrev sortedExpression : Expr :=
  .builtinCall .sorted [.tuple [.name "a", .name "b", .name "c"]]
private def sortedHeap (heap : Mutable.Heap) (a b c : Int) : Mutable.Heap :=
  heap ++ [.tuple (triple a b c), .list (triple (min a b) (max a b) c)]

/-- Execute the actual tuple display and sorted call, retaining both allocations. -/
theorem sortedExpression.result (env : HeapCalls.Env) (heap : Mutable.Heap) (a b c : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) (ac : a ≤ c) (bc : b ≤ c) :
    HeapCalls.Evaluates program env sortedExpression heap (.ref (heap.length + 1))
      (sortedHeap heap a b c) := by
  have tupleRun := HeapCalls.Evaluates.tuple (globals := []) (module := program) (heap := heap)
    (.cons (HeapCalls.Evaluates.lookup ha) (.cons (HeapCalls.Evaluates.lookup hb)
      (.cons (HeapCalls.Evaluates.lookup hc) .nil))) (by simp)
  apply HeapCalls.Evaluates.builtin (.cons tupleRun .nil)
  have operation := HeapCalls.primitive_sorted_tuple (heap ++ [.tuple (triple a b c)])
    heap.length [.int a, .int b, .int c] [.int (min a b), .int (max a b), .int c]
    (by simp [triple]) (ScalarSort.three_top a b c ac bc)
  simpa [sortedHeap, triple, List.append_assoc] using operation

theorem sortedAssignment.result (env : HeapCalls.Env) (heap : Mutable.Heap) (a b c : Int)
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) (ac : a ≤ c) (bc : b ≤ c) :
    HeapCalls.BlockContinues program env
      [.assign [.tuple [.name "a", .name "b", .name "c"]] sortedExpression] heap
      (HeapCalls.assignNames env ["a", "b", "c"] (triple (min a b) (max a b) c))
      (sortedHeap heap a b c) := by
  apply HeapCalls.BlockContinues.unpack rfl (sortedExpression.result env heap a b c ha hb hc ac bc)
  · simp [Mutable.sequence, sortedHeap, triple]
  · rfl

private abbrev square (name : String) : Expr := .binOp (.name name) .mult (.name name)
private abbrev triangleCondition : Expr := .boolOp .and
  [.compare (.name "a") [(.gt, .constant (.int 0))],
   .compare (.binOp (square "a") .add (square "b")) [(.eq, square "c")],
   .compare (.binOp (.binOp (.name "a") .add (.name "b")) .add (.name "c")) [(.eq, .name "total")]]
private abbrev productExpression : Expr := .binOp (.binOp (.name "a") .mult (.name "b")) .mult (.name "c")
private abbrev updateBest : List Stmt :=
  [.ifThen triangleCondition [.assign [.name "best"] (.builtinCall .max [.name "best", productExpression])] []]

theorem triangleCondition.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total a b c : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c)))
    (positive : 0 < a) (squares : a * a + b * b = c * c) (perimeter : a + b + c = total) :
    HeapCalls.Evaluates program env triangleCondition heap (.scalar (.bool true)) heap := by
  have first : HeapCalls.Evaluates program env (.compare (.name "a") [(.gt, .constant (.int 0))])
      heap (.scalar (.bool true)) heap := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup ha) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, positive]
  have second : HeapCalls.Evaluates program env
      (.compare (.binOp (square "a") .add (square "b")) [(.eq, square "c")])
      heap (.scalar (.bool true)) heap := by
    apply HeapCalls.Evaluates.compare
      (HeapCalls.Evaluates.binary
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup ha) (HeapCalls.Evaluates.lookup ha) rfl)
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hb) (HeapCalls.Evaluates.lookup hb) rfl) rfl)
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hc) (HeapCalls.Evaluates.lookup hc) rfl)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, squares]
  have third : HeapCalls.Evaluates program env
      (.compare (.binOp (.binOp (.name "a") .add (.name "b")) .add (.name "c")) [(.eq, .name "total")])
      heap (.scalar (.bool true)) heap := by
    apply HeapCalls.Evaluates.compare
      (HeapCalls.Evaluates.binary
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup ha) (HeapCalls.Evaluates.lookup hb) rfl)
        (HeapCalls.Evaluates.lookup hc) rfl) (HeapCalls.Evaluates.lookup ht)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, perimeter]
  exact HeapCalls.Evaluates.boolean_next (flag := true) first rfl rfl
    (HeapCalls.Evaluates.boolean_next (flag := true) second rfl rfl
      (HeapCalls.Evaluates.boolean_last (flag := true) third rfl))

theorem updateBest.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total a b c best : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (ha : env "a" = some (.scalar (.int a))) (hb : env "b" = some (.scalar (.int b)))
    (hc : env "c" = some (.scalar (.int c))) (hbest : env "best" = some (.scalar (.int best)))
    (positive : 0 < a) (squares : a * a + b * b = c * c) (perimeter : a + b + c = total) :
    HeapCalls.BlockContinues program env updateBest heap
      (HeapCalls.setLocal env "best" (.scalar (.int (max best (a * b * c))))) heap := by
  apply HeapCalls.BlockContinues.branch (flag := true)
    (triangleCondition.result env heap total a b c ht ha hb hc positive squares perimeter) rfl
  apply HeapCalls.BlockContinues.assign
  apply HeapCalls.Evaluates.builtin
    (.cons (HeapCalls.Evaluates.lookup hbest) (.cons
      (HeapCalls.Evaluates.binary
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup ha) (HeapCalls.Evaluates.lookup hb) rfl)
        (HeapCalls.Evaluates.lookup hc) rfl) .nil))
  by_cases larger : best < a * b * c
  · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom,
      HeapCalls.scalarResult, scalarCompare, evalCmpOp, larger, max_eq_right (le_of_lt larger)]
  · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom,
      HeapCalls.scalarResult, scalarCompare, evalCmpOp, larger, max_eq_left (le_of_not_gt larger)]

private abbrev kExpression : Expr := .binOp (.name "total") .floorDiv (.name "s")
private abbrev aExpression : Expr := .binOp (.name "k") .mult (.binOp (square "m") .sub (square "n"))
private abbrev bExpression : Expr := .binOp (.name "k") .mult
  (.binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult (.name "n"))
private abbrev cExpression : Expr := .binOp (.name "k") .mult (.binOp (square "m") .add (square "n"))
private abbrev coefficientBody : List Stmt :=
  [.assign [.name "k"] kExpression, .assign [.name "a"] aExpression,
   .assign [.name "b"] bExpression, .assign [.name "c"] cExpression]
private def coefficientEnv (env : HeapCalls.Env) (total m n : Nat) :=
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal
    (HeapCalls.setLocal env "k" (.scalar (.int (euclidK total m n))))
    "a" (.scalar (.int (euclidA total m n)))) "b" (.scalar (.int (euclidB total m n))))
    "c" (.scalar (.int (euclidC total m n)))

theorem kExpression.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n : Nat)
    (ht : env "total" = some (.scalar (.int total)))
    (hs : env "s" = some (.scalar (.int (euclidS m n)))) (positive : 0 < euclidS m n) :
    HeapCalls.Evaluates program env kExpression heap (.scalar (.int (euclidK total m n))) heap := by
  apply HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup ht) (HeapCalls.Evaluates.lookup hs)
  simp [Mutable.binary, scalarBinary, euclidK, ne_of_gt positive, positive]

theorem aExpression.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n : Nat)
    (hk : env "k" = some (.scalar (.int (euclidK total m n))))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) (before : n < m) :
    HeapCalls.Evaluates program env aExpression heap (.scalar (.int (euclidA total m n))) heap := by
  rw [(euclid_cast total m n before).1]
  exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hk)
    (HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hm) (HeapCalls.Evaluates.lookup hm) rfl)
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hn) (HeapCalls.Evaluates.lookup hn) rfl) rfl) rfl

theorem bExpression.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n : Nat)
    (hk : env "k" = some (.scalar (.int (euclidK total m n))))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    HeapCalls.Evaluates program env bExpression heap (.scalar (.int (euclidB total m n))) heap := by
  simp only [euclidB, Nat.cast_mul, Nat.cast_ofNat]
  exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hk)
    (HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.binary HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup hm) rfl)
      (HeapCalls.Evaluates.lookup hn) rfl) rfl

theorem cExpression.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n : Nat)
    (hk : env "k" = some (.scalar (.int (euclidK total m n))))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    HeapCalls.Evaluates program env cExpression heap (.scalar (.int (euclidC total m n))) heap := by
  simp only [euclidC, Nat.cast_mul, Nat.cast_add]
  exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hk)
    (HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hm) (HeapCalls.Evaluates.lookup hm) rfl)
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hn) (HeapCalls.Evaluates.lookup hn) rfl) rfl) rfl

theorem coefficientBody.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n : Nat)
    (ht : env "total" = some (.scalar (.int total)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n)))
    (hs : env "s" = some (.scalar (.int (euclidS m n))))
    (positive : 0 < euclidS m n) (before : n < m) :
    HeapCalls.BlockContinues program env coefficientBody heap (coefficientEnv env total m n) heap := by
  apply HeapCalls.BlockContinues.append (HeapCalls.BlockContinues.assign
    (kExpression.result env heap total m n ht hs positive))
  apply HeapCalls.BlockContinues.append (HeapCalls.BlockContinues.assign
    (aExpression.result _ heap total m n rfl
      (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn) before))
  apply HeapCalls.BlockContinues.append (HeapCalls.BlockContinues.assign
    (bExpression.result _ heap total m n rfl
      (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn)))
  exact HeapCalls.BlockContinues.assign (cExpression.result _ heap total m n rfl
    (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn))

private abbrev sortedStatement : Stmt :=
  .assign [.tuple [.name "a", .name "b", .name "c"]] sortedExpression
private abbrev successBody : List Stmt := coefficientBody ++ [sortedStatement] ++ updateBest
private def sortedEnv (env : HeapCalls.Env) (total m n : Nat) : HeapCalls.Env :=
  HeapCalls.assignNames (coefficientEnv env total m n) ["a", "b", "c"]
    (triple (min (euclidA total m n : Int) (euclidB total m n))
      (max (euclidA total m n : Int) (euclidB total m n)) (euclidC total m n))
private def successEnv (env : HeapCalls.Env) (total m n best : Nat) :=
  HeapCalls.setLocal (sortedEnv env total m n) "best" (.scalar (.int (max best (candidate total m n) : Nat)))

theorem successBody.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n best : Nat)
    (ht : env "total" = some (.scalar (.int total)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n)))
    (hs : env "s" = some (.scalar (.int (euclidS m n))))
    (hbest : env "best" = some (.scalar (.int best)))
    (totalPositive : 0 < total) (positive : 0 < n) (before : n < m)
    (divides : total % euclidS m n = 0) :
    HeapCalls.BlockContinues program env successBody heap (successEnv env total m n best)
      (sortedHeap heap (euclidA total m n) (euclidB total m n) (euclidC total m n)) := by
  have mpositive : 0 < m := lt_trans positive before
  have spos : 0 < euclidS m n := by unfold euclidS; positivity
  have coefficients := coefficientBody.result env heap total m n ht hm hn hs spos before
  have squares := euclid_squares total m n before
  have ac : (euclidA total m n : Int) ≤ euclidC total m n := by
    exact_mod_cast (show euclidA total m n ≤ euclidC total m n by nlinarith)
  have bc : (euclidB total m n : Int) ≤ euclidC total m n := by
    exact_mod_cast (show euclidB total m n ≤ euclidC total m n by nlinarith)
  have ordered := sortedAssignment.result (coefficientEnv env total m n) heap
    (euclidA total m n) (euclidB total m n) (euclidC total m n) rfl rfl rfl ac bc
  obtain ⟨candidateProduct, minPositive, perimeter⟩ := euclid_candidate total m n totalPositive positive before divides
  have sortedSquares : min (euclidA total m n) (euclidB total m n) * min (euclidA total m n) (euclidB total m n) +
      max (euclidA total m n) (euclidB total m n) * max (euclidA total m n) (euclidB total m n) =
      euclidC total m n * euclidC total m n := by
    rcases le_total (euclidA total m n) (euclidB total m n) with h | h
    · simpa [min_eq_left h, max_eq_right h] using squares
    · simpa [min_eq_right h, max_eq_left h, Nat.add_comm] using squares
  have product : min (euclidA total m n : Int) (euclidB total m n) *
      max (euclidA total m n : Int) (euclidB total m n) * euclidC total m n = candidate total m n := by
    rw [min_mul_max, candidateProduct]
    simp
  have updated := updateBest.result (sortedEnv env total m n)
    (sortedHeap heap (euclidA total m n) (euclidB total m n) (euclidC total m n))
    total (min (euclidA total m n : Int) (euclidB total m n))
    (max (euclidA total m n : Int) (euclidB total m n)) (euclidC total m n) best
    (by simpa [sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using ht)
    rfl rfl rfl
    (by simpa [sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using hbest)
    (by exact_mod_cast minPositive) (by exact_mod_cast sortedSquares) (by exact_mod_cast perimeter)
  have composed := HeapCalls.BlockContinues.append (HeapCalls.BlockContinues.append coefficients ordered) updated
  simpa only [successBody, sortedStatement, successEnv, sortedEnv, product, Nat.cast_max] using composed

private abbrev sExpression : Expr :=
  .binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult (.binOp (.name "m") .add (.name "n"))
private abbrev skipCondition : Expr :=
  .compare (.binOp (.name "total") .mod (.name "s")) [(.notEq, .constant (.int 0))]
private abbrev innerBody : List Stmt :=
  [.assign [.name "s"] sExpression, .ifThen skipCondition [.continueLoop] []] ++ successBody

theorem sExpression.result (env : HeapCalls.Env) (heap : Mutable.Heap) (m n : Nat)
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    HeapCalls.Evaluates program env sExpression heap (.scalar (.int (euclidS m n))) heap := by
  simp only [euclidS, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat]
  exact HeapCalls.Evaluates.binary
    (HeapCalls.Evaluates.binary HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup hm) rfl)
    (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hm) (HeapCalls.Evaluates.lookup hn) rfl) rfl

theorem skipCondition.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n : Nat)
    (ht : env "total" = some (.scalar (.int total)))
    (hs : env "s" = some (.scalar (.int (euclidS m n)))) (positive : 0 < euclidS m n) :
    HeapCalls.Evaluates program env skipCondition heap
      (.scalar (.bool (decide (total % euclidS m n ≠ 0)))) heap := by
  have remainder : HeapCalls.Evaluates program env (.binOp (.name "total") .mod (.name "s"))
      heap (.scalar (.int (total % euclidS m n))) heap := by
    apply HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup ht) (HeapCalls.Evaluates.lookup hs)
    simp [Mutable.binary, scalarBinary, positive, ne_of_gt positive]
  apply HeapCalls.Evaluates.compare remainder HeapCalls.Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp, Int.natCast_dvd_natCast, Nat.dvd_iff_mod_eq_zero]

theorem innerRange.result (total m n best : Nat) (totalPositive : 0 < total) (positive : 0 < n)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m)))
    (hbest : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv finalHeap, HeapCalls.BlockContinues program env
      [.rangeStepLoop "n" n m 1 innerBody []] heap finalEnv finalHeap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "m" = some (.scalar (.int m)) ∧
      finalEnv "best" = some (.scalar (.int (max best (loopN total m n) : Nat))) := by
  by_cases before : n < m
  · let nextEnv := HeapCalls.setLocal env "n" (.scalar (.int n))
    let withS := HeapCalls.setLocal nextEnv "s" (.scalar (.int (euclidS m n)))
    have nt : nextEnv "total" = some (.scalar (.int total)) := by simpa [nextEnv, HeapCalls.setLocal] using ht
    have nm : nextEnv "m" = some (.scalar (.int m)) := by simpa [nextEnv, HeapCalls.setLocal] using hm
    have st : withS "total" = some (.scalar (.int total)) := by simpa [withS, HeapCalls.setLocal] using nt
    have sm : withS "m" = some (.scalar (.int m)) := by simpa [withS, HeapCalls.setLocal] using nm
    have sbest : withS "best" = some (.scalar (.int best)) := by simpa [withS, nextEnv, HeapCalls.setLocal] using hbest
    have mpositive : 0 < m := lt_trans positive before
    have spos : 0 < euclidS m n := by unfold euclidS; positivity
    have assigned := HeapCalls.BlockContinues.assign (name := "s") (sExpression.result nextEnv heap m n nm rfl)
    have guard := skipCondition.result withS heap total m n st rfl spos
    have running : Calls.rangeContinues (n : Int) m 1 = true := by simp [Calls.rangeContinues, before]
    by_cases divides : total % euclidS m n = 0
    · have success := successBody.result withS heap total m n best st sm rfl rfl sbest totalPositive positive before divides
      have ft : (successEnv withS total m n best) "total" = some (.scalar (.int total)) := by
        simpa [successEnv, sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using st
      have fm : (successEnv withS total m n best) "m" = some (.scalar (.int m)) := by
        simpa [successEnv, sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using sm
      obtain ⟨finalEnv, finalHeap, rest, keptTotal, keptM, answer⟩ := innerRange.result total m (n + 1)
        (max best (candidate total m n)) totalPositive (by omega) (successEnv withS total m n best)
        (sortedHeap heap (euclidA total m n) (euclidB total m n) (euclidC total m n)) ft fm rfl
      refine ⟨finalEnv, finalHeap, ?_, keptTotal, keptM, ?_⟩
      · apply HeapCalls.BlockContinues.range_step running
        apply HeapCalls.BlockContinues.append assigned
        have continuation := HeapCalls.BlockContinues.append success (HeapCalls.BlockContinues.loop_resume rest)
        have falseGuard : HeapCalls.Evaluates program withS skipCondition heap (.scalar (.bool false)) heap := by
          simpa [divides] using guard
        intro following anchors stack
        exact .next rfl ((falseGuard anchors _).trans (.next rfl (by
          simpa only [Nat.cast_add, Nat.cast_one] using continuation following anchors stack)))
      · have unfoldLoop : loopN total m n = max (candidate total m n) (loopN total m (n + 1)) := by
          rw [loopN.eq_def]
          simp only [show ¬n ≥ m by omega, if_false]
          rfl
        simpa only [unfoldLoop, max_assoc] using answer
    · obtain ⟨finalEnv, finalHeap, rest, keptTotal, keptM, answer⟩ := innerRange.result total m (n + 1)
        best totalPositive (by omega) withS heap st sm sbest
      refine ⟨finalEnv, finalHeap, ?_, keptTotal, keptM, ?_⟩
      · apply HeapCalls.BlockContinues.range_step running
        apply HeapCalls.BlockContinues.append assigned
        have trueGuard : HeapCalls.Evaluates program withS skipCondition heap (.scalar (.bool true)) heap := by
          simpa [divides] using guard
        intro following anchors stack
        refine .next rfl ((trueGuard anchors _).trans (.next rfl ?_))
        apply HeapCalls.Reaches.next (following :=
          ⟨.block (.rangeStepLoop "n" ((n : Int) + 1) m 1 innerBody [] :: following) withS, stack, heap⟩) rfl
        simpa only [Nat.cast_add, Nat.cast_one] using rest following anchors stack
      · have unfoldLoop : loopN total m n = loopN total m (n + 1) := by
          rw [loopN.eq_def]
          simp [show ¬n ≥ m by omega, euclidS] at divides ⊢
          simp [divides]
        simpa only [unfoldLoop] using answer
  · refine ⟨env, heap, HeapCalls.BlockContinues.range_done ?_ HeapCalls.BlockContinues.nil, ht, hm, ?_⟩
    · simp [Calls.rangeContinues, before]
    · simpa [loopN.eq_def, show n ≥ m by omega] using hbest
termination_by m - n
decreasing_by all_goals omega

private abbrev outerCondition : Expr := .compare
  (.binOp (.binOp (.constant (.int 2)) .mult (.name "m")) .mult
    (.binOp (.name "m") .add (.constant (.int 1)))) [(.ltE, .name "total")]
private abbrev outerBody : List Stmt :=
  [.forIn (.name "n") (.call (.name "range") [.constant (.int 1), .name "m"]) innerBody [],
   .augAssign (.name "m") .add (.constant (.int 1))]
private abbrev outerLoop : Stmt := .whileDo outerCondition outerBody []

theorem outerCondition.result (env : HeapCalls.Env) (heap : Mutable.Heap) (total m : Nat)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m))) :
    HeapCalls.Evaluates program env outerCondition heap
      (.scalar (.bool (decide (2 * m * (m + 1) ≤ total)))) heap := by
  apply HeapCalls.Evaluates.compare
    (HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.binary HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup hm) rfl)
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup hm) HeapCalls.Evaluates.constant rfl) rfl)
    (HeapCalls.Evaluates.lookup ht)
  have castBound : (2 : Int) * m * ((m : Int) + 1) = (2 * m * (m + 1) : Nat) := by push_cast; rfl
  simp only [HeapCalls.compare, HeapCalls.compareScalars, scalarCompare, evalCmpOp]
  rw [castBound]
  simp
  rw [castBound]
  exact Int.ofNat_le

theorem outerLoop.result (total m best : Nat) (mpositive : 2 ≤ m)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m)))
    (hbest : env "best" = some (.scalar (.int best))) :
    ∃ finalEnv finalHeap, HeapCalls.BlockContinues program env [outerLoop] heap finalEnv finalHeap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧
      finalEnv "best" = some (.scalar (.int (max best (loopM total m) : Nat))) := by
  have guard := outerCondition.result env heap total m ht hm
  by_cases within : 2 * m * (m + 1) ≤ total
  · have totalPositive : 0 < total := by nlinarith
    obtain ⟨rowEnv, rowHeap, row, rowTotal, rowM, rowBest⟩ :=
      innerRange.result total m 1 best totalPositive (by omega) env heap ht hm hbest
    have ranged : HeapCalls.BlockContinues program env
        [.forIn (.name "n") (.call (.name "range") [.constant (.int 1), .name "m"]) innerBody []]
        heap rowEnv rowHeap :=
      HeapCalls.BlockContinues.for_range (.cons HeapCalls.Evaluates.constant
        (.cons (HeapCalls.Evaluates.lookup hm) .nil)) rfl row
    let nextEnv := HeapCalls.setLocal rowEnv "m" (.scalar (.int (m + 1 : Nat)))
    have incremented : HeapCalls.BlockContinues program rowEnv
        [.augAssign (.name "m") .add (.constant (.int 1))] rowHeap nextEnv rowHeap := by
      apply HeapCalls.BlockContinues.augAssign rowM HeapCalls.Evaluates.constant
      simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
    have nextTotal : nextEnv "total" = some (.scalar (.int total)) := by
      simpa [nextEnv, HeapCalls.setLocal] using rowTotal
    have nextBest : nextEnv "best" = some (.scalar (.int (max best (loopN total m 1) : Nat))) := by
      simpa [nextEnv, HeapCalls.setLocal] using rowBest
    obtain ⟨finalEnv, finalHeap, rest, keptTotal, answer⟩ := outerLoop.result total (m + 1)
      (max best (loopN total m 1)) (by omega) nextEnv rowHeap nextTotal rfl nextBest
    refine ⟨finalEnv, finalHeap, ?_, keptTotal, ?_⟩
    · apply HeapCalls.BlockContinues.while_loop (flag := true) (value := .scalar (.bool true))
      · simpa [within] using guard
      · rfl
      · exact HeapCalls.BlockContinues.append (HeapCalls.BlockContinues.append ranged incremented) rest
    · have unfoldLoop : loopM total m = max (loopN total m 1) (loopM total (m + 1)) := by
        rw [loopM.eq_def]
        rw [dif_neg (by omega)]
      simpa only [unfoldLoop, max_assoc] using answer
  · refine ⟨env, heap, ?_, ht, ?_⟩
    · apply HeapCalls.BlockContinues.while_loop (flag := false) (value := .scalar (.bool false))
      · simpa [within] using guard
      · rfl
      · exact HeapCalls.BlockContinues.nil
    · have stopped : loopM total m = 0 := by
        rw [loopM.eq_def]
        rw [dif_pos (by omega)]
      simpa only [stopped, max_zero] using hbest
termination_by total + 1 - m
decreasing_by
  have bound : m ≤ total := by nlinarith
  omega

/-- The actual Python loop returns the established specification on its successful domain. -/
theorem solve.naturals (total : Nat) (heap : Mutable.Heap)
    (positive : 0 < ProjectEulerStatements.P9.naive total) :
    ∃ finalHeap, HeapCalls.ReturnsValue program "solve" [.scalar (.int total)] heap
      (.scalar (.int (ProjectEulerStatements.P9.naive total))) finalHeap := by
  let initialEnv := HeapCalls.setLocal (HeapCalls.setLocal
    (HeapCalls.argumentEnv [⟨"total", some (.name "int"), none⟩] [.scalar (.int total)])
    "best" (.scalar (.int 0))) "m" (.scalar (.int 2))
  obtain ⟨finalEnv, finalHeap, loop, _, answer⟩ := outerLoop.result total 2 0
    (by omega) initialEnv heap rfl rfl rfl
  have output : finalEnv "best" = some (.scalar (.int (ProjectEulerStatements.P9.naive total))) := by
    rw [ProjectEulerSolutions.P9.equiv]
    simpa only [Nat.zero_max, ProjectEulerSolutions.P9.solve] using answer
  refine ⟨finalHeap, ?_⟩
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.assign (HeapCalls.Evaluates.constant (value := .int 0))
  apply HeapCalls.BlockReturns.assign (HeapCalls.Evaluates.constant (value := .int 2))
  apply HeapCalls.BlockContinues.returns loop
  apply HeapCalls.BlockReturns.branch (flag := true) (HeapCalls.Evaluates.lookup output)
  · simp [Mutable.truth, scalarTruth, ne_of_gt positive]
  · exact HeapCalls.BlockReturns.return_value (HeapCalls.Evaluates.lookup output)

theorem solve.«Equiv(naive)» (total : Int) (domain : nonnegative total) (existsTriplet : hasTriplet total) :
    (HeapCalls.scalarFunction program "solve" [.int total]).Equiv
      (HeapCalls.scalarFunction P9Ast.module "naive" [.int total]) := by
  have nonneg := (nonnegative_iff total).mp domain
  lift total to Nat using nonneg
  have positive := (hasTriplet_iff total).mp existsTriplet
  obtain ⟨heap, executed⟩ := solve.naturals total [] positive
  have left := HeapCalls.ReturnsValue.scalar_finishes (values := [.int total]) executed
  have right := HeapCalls.ReturnsValue.scalar_finishes (values := [.int total])
    (data.«project-euler-statements».data.py.project_euler_statements.p9.naive.naturals total [])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem solve.«Supported» :
    (∀ total : Int, nonnegative total → hasTriplet total →
      (HeapCalls.scalarFunction program "solve" [.int total]).Supported) ∧
    (∀ total : Int, nonnegative total → hasTriplet total →
      (HeapCalls.scalarFunction P9Ast.module "naive" [.int total]).Supported) := by
  constructor
  · intro total domain existsTriplet
    have nonneg := (nonnegative_iff total).mp domain
    lift total to Nat using nonneg
    obtain ⟨heap, executed⟩ := solve.naturals total [] ((hasTriplet_iff total).mp existsTriplet)
    exact (HeapCalls.ReturnsValue.scalar_returns (values := [.int total]) executed).supported
  · intro total domain _
    have nonneg := (nonnegative_iff total).mp domain
    lift total to Nat using nonneg
    exact (HeapCalls.ReturnsValue.scalar_returns (values := [.int total])
      (data.«project-euler-statements».data.py.project_euler_statements.p9.naive.naturals total [])).supported

end solvers.«9»
