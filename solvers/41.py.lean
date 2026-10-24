import «.formalsw».solvers.«41.py.ast»
import lang.py.FormalSW.HeapCalls
import data.«project-euler-statements».data.py.project_euler_statements.«p41.py»
import ProjectEulerSolutions.Lib.SixStepPrimality
import ProjectEulerSolutions.Lib.PandigitalPrimeSearch
import ProjectEulerSolutions.Lib.HeapCollection

set_option Elab.async false

namespace solvers.«41»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.SixStepPrimality
private abbrev program := Solvers.P41Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

private theorem boolean_or (left right : Expr) (a b : Bool) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (lhs : Evaluates program env left heap (flag a) heap)
    (rhs : Evaluates program env right heap (flag b) heap) :
    Evaluates program env (.boolOp .or [left, right]) heap (flag (a || b)) heap := by
  cases a with
  | false => exact Evaluates.boolean_next lhs rfl rfl (Evaluates.boolean_last rhs rfl)
  | true => exact Evaluates.boolean_stop lhs rfl rfl

namespace is_prime
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[4]? : Option Stmt) with | some (.whileDo _ body _) => body | _ => []
private abbrev condition : Expr := .compare (.binOp (.name "i") .mult (.name "i")) [(.ltE, .name "n")]

private theorem tested (n first : Nat) (large : 5 ≤ first) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getI : env "i" = some (number first)) :
    Evaluates program env
      (.boolOp .or [
        .compare (.binOp (.name "n") .mod (.name "i")) [(.eq, .constant (.int 0))],
        .compare (.binOp (.name "n") .mod (.binOp (.name "i") .add (.constant (.int 2))))
          [(.eq, .constant (.int 0))]])
      heap (flag (decide (n % first = 0 ∨ n % (first+2) = 0))) heap := by
  have firstTest : Evaluates program env
      (.compare (.binOp (.name "n") .mod (.name "i")) [(.eq, .constant (.int 0))]) heap
      (flag (decide (n % first = 0))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number (n % first)) (bv := number 0)
    · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup getN) (Evaluates.lookup getI)
        (by simp [Mutable.binary, scalarBinary, number, show first ≠ 0 by omega, show 0 < first by omega])
    · exact Evaluates.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast, Nat.dvd_iff_mod_eq_zero]
  have secondTest : Evaluates program env
      (.compare (.binOp (.name "n") .mod (.binOp (.name "i") .add (.constant (.int 2))))
        [(.eq, .constant (.int 0))]) heap (flag (decide (n % (first+2) = 0))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number (n % (first+2))) (bv := number 0)
    · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (av := number n) (bv := number (first+2))
        (Evaluates.lookup getN)
      · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup getI) Evaluates.constant
          (by simp [Mutable.binary, scalarBinary, number])
      · simp [Mutable.binary, scalarBinary, number, show (first+2 : Int) ≠ 0 by omega, show (0 : Int) < first+2 by omega]
    · exact Evaluates.constant
    · change some (Except.ok (ε := ExceptionObject) (flag (decide ((n : Int) % ((first+2 : Nat) : Int) = 0))), heap) = _
      simp only [← Int.natCast_emod, Int.natCast_eq_zero]
  simpa only [Bool.decide_or] using boolean_or _ _ _ _ heap env firstTest secondTest

private theorem loop (n first : Nat) (large : 5 ≤ first) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (getN : env "n" = some (number n))
    (getI : env "i" = some (number first)) :
    BlockReturns program env [.whileDo condition body [], .return (.constant (.bool true))]
      heap (flag (trial n first)) heap := by
  induction measure : n + 1 - first using Nat.strong_induction_on generalizing first env with
  | h measure ih =>
      have compared : Evaluates program env condition heap (flag (decide (first*first ≤ n))) heap := by
        apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number (first*first)) (bv := number n)
        · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup getI) (Evaluates.lookup getI)
            (by simp [Mutable.binary, scalarBinary, number])
        · exact Evaluates.lookup getN
        · change some (Except.ok (ε := ExceptionObject) (flag (decide (((first*first : Nat) : Int) ≤ (n : Int)))), heap) = _
          simp only [Int.ofNat_le]
      rw [trial]
      by_cases active : first*first ≤ n
      · rw [dif_pos ⟨large, active⟩]
        apply BlockReturns.while_loop (middle := heap) (value := flag true) (flag := true) (by simpa only [active, decide_true] using compared) rfl
        by_cases divides : n % first = 0 ∨ n % (first+2) = 0
        · rw [if_pos divides]
          apply BlockReturns.branch (middle := heap) (value := flag true) (flag := true)
            (by simpa only [divides, decide_true] using tested n first large heap env getN getI) rfl
          exact BlockReturns.return_value Evaluates.constant
        · rw [if_neg divides]
          apply BlockReturns.branch (middle := heap) (value := flag false) (flag := false)
            (by simpa only [divides, decide_false] using tested n first large heap env getN getI) rfl
          apply BlockReturns.augAssign (value := number (first+6)) (middle := heap) (afterOperation := heap) getI Evaluates.constant
          · simp [inplace, Mutable.binary, scalarBinary, number]
          · have decreased : n+1-(first+6) < n+1-first := by
              have bounded : first ≤ n := by nlinarith
              omega
            have next := ih (n+1-(first+6)) (by omega)
              (first+6) (by omega) (setLocal env "i" (number (first+6)))
              (by simpa [setLocal] using getN) rfl rfl
            simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
              Bool.false_or, Bool.false_eq_true, if_false, List.nil_append] using next
      · rw [dif_neg (by omega : ¬(5 ≤ first ∧ first*first ≤ n))]
        apply BlockReturns.while_loop (middle := heap) (value := flag false) (flag := false) (by simpa only [active, decide_false] using compared) rfl
        exact BlockReturns.return_value Evaluates.constant

private theorem small (n : Nat) (lower : n < 2) (heap : Mutable.Heap) :
    ReturnsValue program "is_prime" [number n] heap (flag false) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (middle := heap) (value := flag true) (flag := true)
    (Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, lower])) rfl
  exact BlockReturns.return_value Evaluates.constant

private theorem divisible (n p : Nat) (prime : Nat.Prime p) (divides : n % p = 0) :
    Nat.Prime n ↔ n = p := by
  constructor
  · intro primeN
    rcases (Nat.dvd_prime primeN).mp (Nat.dvd_of_mod_eq_zero divides) with one | same
    · have := prime.two_le; omega
    · exact same.symm
  · intro same
    simpa only [same] using prime

/-- The actual Python loop is pure and agrees with primality on every natural input. -/
theorem result (n : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "is_prime" [number n] heap (flag (decide (Nat.Prime n))) heap := by
  by_cases lower : n < 2
  · have bad : ¬ Nat.Prime n := fun prime => by have := prime.two_le; omega
    simpa only [bad, decide_false] using small n lower heap
  let env := functionEnv ((findFunctionIn program.body "is_prime").getD ([], [])).1
    ((findFunctionIn program.body "is_prime").getD ([], [])).2 [number n] []
  have getN : env "n" = some (number n) := rfl
  have modTest (p : Nat) (positive : 0 < p) : Evaluates program env
      (.compare (.binOp (.name "n") .mod (.constant (.int p))) [(.eq, .constant (.int 0))])
      heap (flag (decide (n % p = 0))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number (n % p)) (bv := number 0)
    · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap) (Evaluates.lookup getN) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number, positive, show p ≠ 0 by omega])
    · exact Evaluates.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast, Nat.dvd_iff_mod_eq_zero]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (middle := heap) (value := flag false) (flag := false)
    (Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup getN) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, lower])) rfl
  by_cases even : n % 2 = 0
  · simp only [divisible n 2 Nat.prime_two even]
    apply BlockReturns.branch (middle := heap) (value := flag true) (flag := true) (by simpa only [even, decide_true] using modTest 2 (by decide)) rfl
    apply BlockReturns.return_value
    exact Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup getN) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  · apply BlockReturns.branch (middle := heap) (value := flag false) (flag := false) (by simpa only [even, decide_false] using modTest 2 (by decide)) rfl
    by_cases third : n % 3 = 0
    · simp only [divisible n 3 Nat.prime_three third]
      apply BlockReturns.branch (middle := heap) (value := flag true) (flag := true) (by simpa only [third, decide_true] using modTest 3 (by decide)) rfl
      apply BlockReturns.return_value
      exact Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup getN) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · apply BlockReturns.branch (middle := heap) (value := flag false) (flag := false) (by simpa only [third, decide_false] using modTest 3 (by decide)) rfl
      apply BlockReturns.assign Evaluates.constant
      have searched := loop n 5 (by decide) heap (setLocal env "i" (number 5))
        (by simp [setLocal, getN]) rfl
      simpa only [trial_correct n 5 (by omega) (by decide) (by decide) (initial n even third)] using searched
end is_prime

private abbrev text (digit : Nat) : Mutable.Value := .scalar (.string (toString digit))
private def texts (digits : List Nat) : List Mutable.Value := digits.map text

namespace perm_to_int
private abbrev loopBody : List Stmt :=
  [.assign [.name "x"] (.binOp (.binOp (.name "x") .mult (.constant (.int 10))) .add
    (.binOp (.builtinCall .ord [.name "ch"]) .sub (.constant (.int 48))))]

private theorem ordinal : ∀ digit : Fin 10,
    Strings.ord (.string (toString digit.val)) = .ok (.int (digit.val+48)) := by decide +kernel

private theorem loop (items remaining : List Nat) (position address total : Nat)
    (decimal : ∀ d ∈ remaining, d < 10) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (sequence : Mutable.sequence (.ref address) heap = some (.ok (texts items), heap))
    (suffix : items.drop position = remaining) (getTotal : env "x" = some (number total)) :
    BlockReturns program env
      [.sequenceStepLoop "ch" (.ref address) position loopBody [], .return (.name "x")]
      heap (number (remaining.foldl (fun acc d => acc*10+d) total)) heap := by
  induction remaining generalizing position total env with
  | nil =>
      apply BlockReturns.sequence_done (source := .ref address) sequence
      · have missing : items[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [texts, missing]
      · exact BlockReturns.return_value (Evaluates.lookup getTotal)
  | cons head tail ih =>
      have item : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      let bound := setLocal env "ch" (text head)
      have boundTotal : bound "x" = some (number total) := by simpa [bound, setLocal] using getTotal
      have getCh : bound "ch" = some (text head) := by simp [bound, setLocal]
      have ord : Evaluates program bound (.builtinCall .ord [.name "ch"]) heap
          (number (head+48)) heap := by
        apply Evaluates.builtin (.cons (Evaluates.lookup getCh) .nil)
        simp only [primitive]
        rw [ordinal ⟨head, decimal head (by simp)⟩]
        rfl
      have suffixTail : items.drop (position+1) = tail := by simp [List.drop_add_one_eq_tail_drop, suffix]
      have next := ih (position+1) (total*10+head)
        (fun d member => decimal d (by simp [member]))
        (setLocal bound "x" (number (total*10+head))) suffixTail rfl
      apply BlockReturns.sequence_step (source := .ref address) (value := text head) sequence (by simp [texts, item])
      apply BlockReturns.assign (value := number (total*10+head)) (middle := heap)
      · apply Evaluates.binary (middle := heap) (afterRight := heap)
          (av := number (total*10)) (bv := number head)
        · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
            (Evaluates.lookup boundTotal) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
        · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
            ord Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
        · simp [Mutable.binary, scalarBinary, number]
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false, List.nil_append, List.foldl_cons] using next

/-- Conversion reads the actual tuple characters and preserves their heap. -/
theorem result (digits : List Nat) (decimal : ∀ d ∈ digits, d < 10)
    (heap : Mutable.Heap) (address : Nat)
    (sequence : Mutable.sequence (.ref address) heap = some (.ok (texts digits), heap)) :
    ReturnsValue program "perm_to_int" [.ref address] heap
      (number (ProjectEulerSolutions.PandigitalPrimeSearch.value digits)) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_sequence (items := texts digits) (value := .ref address)
    (by intros; simp) (Evaluates.lookup rfl) sequence
  simpa only [ProjectEulerSolutions.PandigitalPrimeSearch.fold_value, Nat.zero_mul, Nat.zero_add]
    using loop digits digits 0 address 0 decimal heap _ sequence rfl rfl
end perm_to_int

namespace solve
open ProjectEulerSolutions.PandigitalPrimeSearch ProjectEulerSolutions.HeapCollection

private theorem pool_result (width : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number width)) :
    ∃ extra, Evaluates program env
      (.listComp (.builtinCall .str [.name "d"]) [("d", .call (.name "range")
        [.constant (.int 1), .binOp (.name "n") .add (.constant (.int 1))], [])])
      heap (.ref heap.length) (working heap (texts (pool width)) extra) := by
  apply range_map program "d" (.builtinCall .str [.name "d"]) text
    (values := [number 1, number (width+1)]) (first := 1) (count := width)
  · intro heap env n found
    refine ⟨[], ?_⟩
    simpa only [List.append_nil] using
      (Evaluates.builtin (module := program) (env := env) (heap := heap)
        (.cons (Evaluates.lookup found) .nil) (by rfl) :
        Evaluates program env (.builtinCall .str [.name "d"]) heap (text n) heap)
  · exact .cons Evaluates.constant (.cons
      (Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
        (av := number width) (bv := number 1) (value := number (width+1))
        (Evaluates.lookup getN) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number])) .nil)
  · change Except.ok (ε := ExceptionObject) ((1 : Int), ((width+1 : Nat) : Int), (1 : Int)) =
      .ok (1, (1 : Int) + width, 1)
    simp only [Int.natCast_add, Int.natCast_one, Int.add_comm]

/-- The comparison and primality call use Python's short-circuit evaluation order. -/
private theorem better_result (current candidate : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getBest : env "best" = some (number current)) (getVal : env "val" = some (number candidate)) :
    Evaluates program env
      (.boolOp .and [.compare (.name "val") [(.gt, .name "best")],
        .call (.name "is_prime") [.name "val"]])
      heap (flag (decide (current < candidate ∧ Nat.Prime candidate))) heap := by
  have compared : Evaluates program env (.compare (.name "val") [(.gt, .name "best")])
      heap (flag (decide (current < candidate))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap)
      (Evaluates.lookup getVal) (Evaluates.lookup getBest)
    change some (Except.ok (ε := ExceptionObject) (flag (decide ((current : Int) < (candidate : Int)))), heap) = _
    simp only [Int.ofNat_lt]
  by_cases better : current < candidate
  · have tested := Evaluates.call (module := program) (env := env)
      (.cons (Evaluates.lookup getVal) .nil) (is_prime.result candidate heap)
    simpa only [better, true_and, decide_true] using
      Evaluates.boolean_next (by simpa only [better, decide_true] using compared) rfl rfl
        (Evaluates.boolean_last tested rfl)
  · simpa only [better, false_and, decide_false] using
      Evaluates.boolean_stop (by simpa only [better, decide_false] using compared) rfl rfl

private abbrev code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private abbrev outerBody : List Stmt :=
  match (code[0]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private abbrev permutationBody : List Stmt :=
  match (outerBody[4]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []

/-- One source iteration reaches its ordinary loop boundary, possibly through continue. -/
private def Iteration (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap) : Prop :=
  ∀ next following anchors stack, Reaches program anchors
    ⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩
    ⟨.block (next :: following) finalEnv, stack, finalHeap⟩

private theorem Iteration.done (env : HeapCalls.Env) (heap : Mutable.Heap) :
    Iteration env [] heap env heap := fun _ _ _ _ => .next rfl .refl

private theorem Iteration.prepend
    (first : BlockContinues program env body heap middleEnv middleHeap)
    (rest : Iteration middleEnv tail middleHeap finalEnv finalHeap) :
    Iteration env (body ++ tail) heap finalEnv finalHeap := by
  intro next following anchors stack
  simpa only [List.append_assoc, List.singleton_append] using
    (first (tail ++ (.loopResume next :: following)) anchors stack).trans
      (by simpa only [List.append_assoc, List.singleton_append] using rest next following anchors stack)

private theorem Iteration.branch {value : Mutable.Value} {flag : Bool}
    (evaluated : Evaluates program env condition heap value middle)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : Iteration env ((if flag then yes else no) ++ body) middle finalEnv finalHeap) :
    Iteration env (.ifThen condition yes no :: body) heap finalEnv finalHeap := by
  intro next following anchors stack
  apply Reaches.next rfl
  apply (evaluated anchors _).trans
  apply Reaches.next (following :=
    ⟨.block (((if flag then yes else no) ++ body) ++ [.loopResume next] ++ following) env, stack, middle⟩)
    (by simp [HeapCalls.step, run, truth, List.append_assoc])
  exact rest next following anchors stack

private theorem Iteration.skip
    (target : ∀ next following,
      Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following)) :
    Iteration env (.continueLoop :: body) heap env heap := by
  intro next following anchors stack
  have found : Calls.loopTarget (body ++ (.loopResume next :: following)) = some (next, following) := by
    simpa only [List.append_assoc, List.singleton_append] using target next following
  exact .next (by simp [HeapCalls.step, found]) .refl

private theorem Iteration.continues
    (iteration : Iteration env body heap middleEnv middleHeap)
    (rest : BlockContinues program middleEnv [next] middleHeap finalEnv finalHeap) :
    BlockContinues program env (body ++ [.loopResume next]) heap finalEnv finalHeap := by
  intro following anchors stack
  exact (iteration next following anchors stack).trans (rest following anchors stack)

private theorem tuple_last (digits : List Nat) (nonempty : digits ≠ []) (heap : Mutable.Heap)
    (address : Nat) (found : heap[address]? = some (.tuple (texts digits))) :
    Mutable.getItem (.ref address) (.scalar (.int (-1))) heap =
      some (.ok (text (lastDigit digits)), heap) := by
  have positive : 0 < digits.length := List.length_pos_iff.mpr nonempty
  have notLow : ¬ (-1 + (digits.length : Int) < 0) := by omega
  have notHigh : ¬ (-1 + (digits.length : Int) ≥ digits.length) := by omega
  have cast : (-1 + (digits.length : Int)).toNat = digits.length - 1 := by omega
  have index : Mutable.index (texts digits) (.scalar (.int (-1))) heap =
      some (.ok (digits.length - 1), heap) := by
    simp [Mutable.index, texts, notLow, notHigh, cast]
  have selected : ((texts digits)[digits.length-1]?).getD (.scalar .none) = text (lastDigit digits) := by
    simp only [texts, List.getElem?_map, ← List.getLast?_eq_getElem?, List.getLast?_eq_head?_reverse]
    cases reversed : digits.reverse with
    | nil => exact False.elim (nonempty (List.reverse_eq_nil_iff.mp reversed))
    | cons head tail => simp [lastDigit, reversed]
  simp [Mutable.getItem, found, index, Except.map, selected]

private theorem last_result (digits : List Nat) (nonempty : digits ≠ []) (heap : Mutable.Heap)
    (address : Nat) (env : HeapCalls.Env) (getPerm : env "perm" = some (.ref address))
    (found : heap[address]? = some (.tuple (texts digits))) :
    Evaluates program env (.subscript (.name "perm") (.unaryOp .uSub (.constant (.int 1))))
      heap (text (lastDigit digits)) heap := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getPerm)
    (Evaluates.unary Evaluates.constant rfl)
  exact tuple_last digits nonempty heap address found

private theorem membership_digit (digit : Nat) (bounded : digit < 10) (heap : Mutable.Heap)
    (address : Nat) (found : heap[address]? = some (.tuple (texts [2,4,5,6,8]))) :
    HeapCalls.membership (text digit) (.ref address) false heap =
      some (.ok (flag (decide (digit ∈ [2,4,5,6,8]))), heap) := by
  interval_cases digit <;> simp [HeapCalls.membership, found] <;> rfl

private abbrev forbidden : Expr := .tuple [.constant (.string "2"), .constant (.string "4"),
  .constant (.string "5"), .constant (.string "6"), .constant (.string "8")]
private abbrev filterHeap : Mutable.Heap := [.tuple (texts [2,4,5,6,8])]

private theorem filter_result (digit : Nat) (bounded : digit < 10) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (getLast : env "last" = some (text digit)) :
    Evaluates program env (.membership (.name "last") forbidden false) heap
      (flag (decide (digit ∈ [2,4,5,6,8]))) (heap ++ filterHeap) := by
  have tupled : Evaluates program env forbidden heap (.ref heap.length) (heap ++ filterHeap) :=
    Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant
      (.cons Evaluates.constant (.cons Evaluates.constant .nil))))) (by decide)
  apply Evaluates.membership (.cons (Evaluates.lookup getLast) (.cons tupled .nil))
  exact membership_digit digit bounded (heap ++ filterHeap) heap.length (by simp [filterHeap])

private theorem permutation_iteration (digits : List Nat) (nonempty : digits ≠ [])
    (decimal : ∀ d ∈ digits, d < 10) (current : Nat) (heap : Mutable.Heap) (address : Nat)
    (env : HeapCalls.Env) (getPerm : env "perm" = some (.ref address))
    (getBest : env "best" = some (number current))
    (found : heap[address]? = some (.tuple (texts digits))) :
    ∃ finalEnv, Iteration env permutationBody heap finalEnv (heap ++ filterHeap) ∧
      finalEnv "best" = some (number (select current digits)) := by
  let lastEnv := setLocal env "last" (text (lastDigit digits))
  have last := BlockContinues.assign (name := "last")
    (last_result digits nonempty heap address env getPerm found)
  have lastBound : lastDigit digits < 10 := by
    have member : lastDigit digits ∈ digits := by
      rw [lastDigit]
      have notNil : digits.reverse ≠ [] := by simpa using nonempty
      apply List.mem_reverse.mp
      cases reversed : digits.reverse with
      | nil => exact False.elim (notNil reversed)
      | cons head tail => simp
    exact decimal _ member
  have filtered := filter_result (lastDigit digits) lastBound heap lastEnv rfl
  have savedBest : lastEnv "best" = some (number current) := by simpa [lastEnv, setLocal] using getBest
  by_cases rejected : lastDigit digits ∈ [2,4,5,6,8]
  · refine ⟨lastEnv, ?_, ?_⟩
    · apply Iteration.prepend last
      apply Iteration.branch (flag := true) filtered
      · simp [Mutable.truth, scalarTruth, rejected]
      · exact Iteration.skip (by intros; rfl)
    · simpa [select, allowed, rejected] using savedBest
  · let finalHeap := heap ++ filterHeap
    let valEnv := setLocal lastEnv "val" (number (value digits))
    have savedPerm : lastEnv "perm" = some (.ref address) := by simpa [lastEnv, setLocal] using getPerm
    have sequence : Mutable.sequence (.ref address) finalHeap = some (.ok (texts digits), finalHeap) := by
      have entry := prefix_found heap filterHeap address _ found
      simp [Mutable.sequence, finalHeap, entry]
    have converted := BlockContinues.assign (name := "val")
      (Evaluates.call (.cons (Evaluates.lookup savedPerm) .nil)
        (perm_to_int.result digits decimal finalHeap address sequence))
    have compared := better_result current (value digits) finalHeap valEnv
      (by simpa [valEnv, setLocal] using savedBest) rfl
    by_cases better : current < value digits ∧ Nat.Prime (value digits)
    · let bestEnv := setLocal valEnv "best" (number (value digits))
      refine ⟨bestEnv, ?_, ?_⟩
      · apply Iteration.prepend last
        apply Iteration.branch (flag := false) filtered
        · simp [Mutable.truth, scalarTruth, rejected]
        · apply Iteration.prepend converted
          apply Iteration.branch (flag := true) compared
          · simp [Mutable.truth, scalarTruth, better]
          · exact Iteration.prepend (BlockContinues.assign (name := "best") (Evaluates.lookup rfl))
              (Iteration.done bestEnv finalHeap)
      · simp [bestEnv, select, allowed, rejected, better, setLocal]
    · refine ⟨valEnv, ?_, ?_⟩
      · apply Iteration.prepend last
        apply Iteration.branch (flag := false) filtered
        · simp [Mutable.truth, scalarTruth, rejected]
        · apply Iteration.prepend converted
          apply Iteration.branch (flag := false) compared
          · simp [Mutable.truth, scalarTruth, better]
          · exact Iteration.done valEnv finalHeap
      · simpa [select, allowed, rejected, better, valEnv, setLocal] using savedBest

private def rowSources (digits : List Nat) : List SequenceSource :=
  digits.map (fun (digit : Nat) => .scalar (.string (toString digit)))

private theorem permutation_continuation (next : Stmt) :
    Calls.loopContinuation permutationBody next = .loopResume next := rfl

private theorem permutation_loop (remaining : List (List Nat))
    (valid : ∀ digits ∈ remaining, digits ≠ [] ∧ ∀ d ∈ digits, d < 10)
    (current : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getBest : env "best" = some (number current)) :
    ∃ finalEnv extra, BlockContinues program env
      [.permutationStepLoop "perm" (remaining.map rowSources) permutationBody []]
      heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number (remaining.foldl select current)) := by
  induction remaining generalizing current heap env with
  | nil =>
      refine ⟨env, [], ?_, getBest⟩
      simpa only [List.map_nil, List.append_nil] using
        (BlockContinues.permutation_done (module := program) (env := env) (heap := heap) .nil)
  | cons head tail ih =>
      obtain ⟨nonempty, decimal⟩ := valid head (by simp)
      obtain ⟨address, allocatedExtra, allocated, found⟩ := allocate_tuple (texts head) heap
      let bound := setLocal env "perm" (.ref address)
      obtain ⟨afterEnv, iteration, afterBest⟩ := permutation_iteration head nonempty decimal
        current (heap ++ allocatedExtra) address bound rfl
        (by simpa [bound, setLocal] using getBest) found
      obtain ⟨finalEnv, restExtra, rest, finalBest⟩ := ih
        (fun digits member => valid digits (by simp [member]))
        (select current head) ((heap ++ allocatedExtra) ++ filterHeap) afterEnv afterBest
      refine ⟨finalEnv, allocatedExtra ++ filterHeap ++ restExtra, ?_, ?_⟩
      · simp only [List.map_cons]
        apply BlockContinues.permutation_step (tuple := .ref address) (middle := heap ++ allocatedExtra)
          (by simpa [rowSources, texts, List.map_map, Function.comp_def, sequenceValue] using allocated)
        simpa only [permutation_continuation, List.map_cons, List.append_assoc] using
          iteration.continues rest
      · simpa only [List.foldl_cons] using finalBest

private theorem width_result (width : Nat) (positive : 0 < width) (bounded : width ≤ 9)
    (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env)
    (getDigits : env "digits" = some (.ref address)) (getBest : env "best" = some (number 0))
    (found : heap[address]? = some (.list (texts (pool width)))) :
    ∃ finalEnv extra, BlockContinues program env [outerBody[4]'(by decide)]
      heap finalEnv (heap ++ extra) ∧
      finalEnv "best" = some (number ((rows width).foldl select 0)) := by
  obtain ⟨finalEnv, extra, executed, best⟩ := permutation_loop (rows width)
    (by
      intro digits member
      have length := row_length width digits member
      exact ⟨by intro empty; simp [empty] at length; omega,
        fun d inside => by have := row_digits width digits member d inside; omega⟩)
    0 heap env getBest
  refine ⟨finalEnv, extra, ?_, best⟩
  apply BlockContinues.for_permutations
    (.cons (Evaluates.lookup getDigits) (.cons Evaluates.constant .nil))
  · simpa only [texts, List.length_map, pool, List.length_range', rows,
      Permutations.rows_map, List.map_map, Function.comp_def, rowSources, sequenceSource] using
      permutationRows_list heap address (texts (pool width)) found
  · exact executed

private theorem digit_sum_result (width : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number width)) :
    Evaluates program env
      (.binOp (.binOp (.name "n") .mult (.binOp (.name "n") .add (.constant (.int 1))))
        .floorDiv (.constant (.int 2))) heap (number (width*(width+1)/2)) heap := by
  apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
    (av := number (width*(width+1))) (bv := number 2)
  · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
      (av := number width) (bv := number (width+1)) (Evaluates.lookup getN)
    · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
        (Evaluates.lookup getN) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
    · simp [Mutable.binary, scalarBinary, number]
  · exact Evaluates.constant
  · simp [Mutable.binary, scalarBinary, number]

private theorem skip_result (sum : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getSum : env "digit_sum" = some (number sum)) :
    Evaluates program env (.compare (.binOp (.name "digit_sum") .mod (.constant (.int 3)))
      [(.eq, .constant (.int 0))]) heap (flag (decide (sum%3=0))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number (sum%3)) (bv := number 0)
  · exact Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup getSum) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
  · exact Evaluates.constant
  · change some (Except.ok (ε := ExceptionObject) (flag (decide (((sum%3 : Nat) : Int)=0))), heap) = _
    simp only [Int.natCast_eq_zero]

private theorem nonzero_result (best : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getBest : env "best" = some (number best)) :
    Evaluates program env (.compare (.name "best") [(.notEq, .constant (.int 0))])
      heap (flag (decide (best ≠ 0))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup getBest) Evaluates.constant
  change some (Except.ok (ε := ExceptionObject) (flag (decide ((best : Int) ≠ 0))), heap) = _
  simp only [ne_eq, Int.natCast_eq_zero]

private theorem outer_continuation (next : Stmt) :
    Calls.loopContinuation outerBody next = .loopResume next := rfl

private theorem outer_loop (width : Nat) (bounded : width ≤ 9) (heap : Mutable.Heap) (env : HeapCalls.Env) :
    ∃ finalHeap, BlockReturns program env
      [.rangeStepLoop "n" width 0 (-1) outerBody [], .return (.constant (.int 0))]
      heap (number (search width)) finalHeap := by
  induction width generalizing heap env with
  | zero =>
      refine ⟨heap, ?_⟩
      apply BlockReturns.range_done (by decide)
      exact BlockReturns.return_value Evaluates.constant
  | succ width ih =>
      let bound := setLocal env "n" (number (width+1))
      let sumEnv := setLocal bound "digit_sum" (number ((width+1)*(width+2)/2))
      have summed := digit_sum_result (width+1) heap bound rfl
      have tested := skip_result ((width+1)*(width+2)/2) heap sumEnv rfl
      have decrement : (width : Int) + 1 + (-1) = width := by omega
      have active : Calls.rangeContinues (width+1) 0 (-1) = true := by simp [Calls.rangeContinues]
      by_cases skipped : ((width+1)*(width+2)/2)%3=0
      · obtain ⟨finalHeap, rest⟩ := ih (by omega) heap sumEnv
        refine ⟨finalHeap, ?_⟩
        rw [search, show optimizedWidth (width+1) = 0 by simp [optimizedWidth, skipped, Nat.add_assoc]]
        simp only [if_true]
        apply BlockReturns.range_step active
        apply BlockReturns.assign summed
        apply BlockReturns.branch (flag := true) tested
        · simp [Mutable.truth, scalarTruth, skipped]
        · apply BlockReturns.continue_loop (by rfl)
          simpa only [outer_continuation, decrement, List.nil_append] using rest
      · obtain ⟨garbage, pooled⟩ := pool_result (width+1) heap sumEnv
          (by simp [sumEnv, bound, setLocal])
        let poolHeap := working heap (texts (pool (width+1))) garbage
        let digitsEnv := setLocal sumEnv "digits" (.ref heap.length)
        let ready := setLocal digitsEnv "best" (number 0)
        obtain ⟨finalEnv, extra, loop, getBest⟩ := width_result (width+1) (by omega) bounded
          poolHeap heap.length ready (by simp [ready, digitsEnv, setLocal]) rfl
          (found heap (texts (pool (width+1))) garbage)
        let finalHeap := poolHeap ++ extra
        let computed := (rows (width+1)).foldl select 0
        have optimized : optimizedWidth (width+1) = computed := by
          simp [optimizedWidth, skipped, computed, Nat.add_assoc]
        have compared := nonzero_result computed finalHeap finalEnv getBest
        by_cases empty : computed = 0
        · obtain ⟨answerHeap, rest⟩ := ih (by omega) finalHeap finalEnv
          refine ⟨answerHeap, ?_⟩
          rw [search, optimized, if_pos empty]
          apply BlockReturns.range_step active
          apply BlockReturns.assign summed
          apply BlockReturns.branch (flag := false) tested
          · simp [Mutable.truth, scalarTruth, skipped]
          · apply BlockReturns.assign pooled
            apply BlockReturns.assign Evaluates.constant
            apply BlockContinues.returns loop
            apply BlockReturns.branch (flag := false) compared
            · simp [Mutable.truth, scalarTruth, empty]
            · apply BlockReturns.loop_resume
              simpa only [outer_continuation, decrement, List.nil_append] using rest
        · refine ⟨finalHeap, ?_⟩
          rw [search, optimized, if_neg empty]
          apply BlockReturns.range_step active
          apply BlockReturns.assign summed
          apply BlockReturns.branch (flag := false) tested
          · simp [Mutable.truth, scalarTruth, skipped]
          · apply BlockReturns.assign pooled
            apply BlockReturns.assign Evaluates.constant
            apply BlockContinues.returns loop
            apply BlockReturns.branch (flag := true) compared
            · simp [Mutable.truth, scalarTruth, empty]
            · exact BlockReturns.return_value (Evaluates.lookup getBest)

/-- The optimized source returns the same maximum as the exhaustive reference. -/
theorem result (width : Nat) (bounded : width ≤ 9) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number width] heap (number (through width)) finalHeap := by
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number width] []
  obtain ⟨finalHeap, executed⟩ := outer_loop width bounded heap initial
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.for_range (values := [number width, number 0, .scalar (.int (-1))])
    (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant
      (.cons (Evaluates.unary Evaluates.constant rfl) .nil))) rfl
  simpa only [search_correct width bounded] using executed

private abbrev inputEnv (max_digits : Int) : ScalarEnv :=
  fun name => if name = "max_digits" then some (.int max_digits) else none

private theorem condition_result (max_digits : Int) :
    Calls.Evaluates program (inputEnv max_digits) Solvers.P41Ast.requires_solve_0
      (.bool (decide (0 ≤ max_digits ∧ max_digits ≤ 9))) := by
  intro stack
  apply Calls.Reaches.next rfl
  apply Calls.Reaches.next rfl
  apply Calls.Reaches.next rfl
  apply Calls.Reaches.next rfl
  by_cases lower : 0 ≤ max_digits
  · simp only [lower, true_and]
    apply Calls.Reaches.next (following :=
      ⟨.expr (.constant (.int 9)) (inputEnv max_digits),
        .chainRight .ltE (.int max_digits) [] (inputEnv max_digits) :: stack⟩)
      (by simp [Calls.step, scalarCompare, evalCmpOp, lower])
    apply Calls.Reaches.next rfl
    by_cases upper : max_digits ≤ 9 <;> simp only [upper, decide_true, decide_false]
    · exact Calls.Reaches.next (by simp [Calls.step, scalarCompare, evalCmpOp, upper]) .refl
    · exact Calls.Reaches.next (by simp [Calls.step, scalarCompare, evalCmpOp, upper]) .refl
  · simp only [lower, false_and, decide_false]
    exact Calls.Reaches.next (by simp [Calls.step, scalarCompare, evalCmpOp, lower]) .refl

private theorem condition_bounds (max_digits : Int)
    (condition : Calls.Condition program (inputEnv max_digits) Solvers.P41Ast.requires_solve_0) :
    0 ≤ max_digits ∧ max_digits ≤ 9 := by
  simpa using (condition_result max_digits).condition_iff.mp condition

theorem «Equiv(naive_bound)» (max_digits : Int)
    (condition : Calls.Condition program (inputEnv max_digits) Solvers.P41Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int max_digits]).Equiv
      (HeapCalls.scalarFunction P41Ast.module "naive_bound" [.int max_digits]) := by
  obtain ⟨nonnegative, bounded⟩ := condition_bounds max_digits condition
  lift max_digits to Nat using nonnegative
  obtain ⟨_, optimizedResult⟩ := result max_digits (by exact_mod_cast bounded) []
  obtain ⟨_, referenceResult⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p41.naive_bound.result max_digits []
  have optimized := optimizedResult.scalar_finishes (values := [.int max_digits])
  have original := referenceResult.scalar_finishes (values := [.int max_digits])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem «Supported» :
    (∀ max_digits : Int, Calls.Condition program (inputEnv max_digits) Solvers.P41Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int max_digits]).Supported) ∧
    (∀ max_digits : Int, Calls.Condition program (inputEnv max_digits) Solvers.P41Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P41Ast.module "naive_bound" [.int max_digits]).Supported) := by
  constructor
  · intro max_digits condition
    obtain ⟨nonnegative, bounded⟩ := condition_bounds max_digits condition
    lift max_digits to Nat using nonnegative
    obtain ⟨_, executed⟩ := result max_digits (by exact_mod_cast bounded) []
    exact (executed.scalar_returns (values := [.int max_digits])).supported
  · intro max_digits condition
    have nonnegative := (condition_bounds max_digits condition).1
    lift max_digits to Nat using nonnegative
    obtain ⟨_, executed⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p41.naive_bound.result max_digits []
    exact (executed.scalar_returns (values := [.int max_digits])).supported
end solve

end solvers.«41»
