import data.«project-euler-statements».data.py.project_euler_statements.«p58.py»
import ProjectEulerSolutions.Lib.SpiralRowCertificates.All
import «.formalsw».solvers.«58.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.HeapMemoCallRules
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.MillerRabin
import ProjectEulerSolutions.Lib.SpiralMillerAgreement

namespace solvers.«58».miller_check
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin

private abbrev program := Solvers.P58Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev lastExpression : Expr := .binOp (.name "n") .sub (.constant (.int 1))
private abbrev lastTest : Expr := .compare (.name "x") [(.eq, lastExpression)]
private abbrev initialTest : Expr := .boolOp .or
  [.compare (.name "x") [(.eq, .constant (.int 1))], lastTest]
private abbrev squareExpression : Expr :=
  .binOp (.binOp (.name "x") .mult (.name "x")) .mod (.name "n")
private abbrev squareBody : List Stmt :=
  [.assign [.name "x"] squareExpression,
   .ifThen lastTest [.return (.constant (.bool true))] []]

private theorem last_test (n : Nat) (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getX : env "x" = some (number x))
    (globals : Mutable.Globals) :
    HeapCalls.Evaluates program env lastTest heap (flag (decide (x = (n : Int) - 1))) heap globals := by
  apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup getX)
    (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getN) HeapCalls.Evaluates.constant rfl)
  simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]

private theorem initial_test (n : Nat) (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getX : env "x" = some (number x))
    (globals : Mutable.Globals) :
    HeapCalls.Evaluates program env initialTest heap
      (flag (decide (x = 1 ∨ x = (n : Int) - 1))) heap globals := by
  have first : HeapCalls.Evaluates program env
      (.compare (.name "x") [(.eq, .constant (.int 1))]) heap
      (flag (decide (x = 1))) heap globals := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup getX) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
  by_cases one : x = 1
  · simp only [one, true_or, decide_true]
    apply HeapCalls.Evaluates.boolean_stop (flag := true)
      (by simpa [one] using first) rfl rfl
  · apply HeapCalls.Evaluates.boolean_next (flag := false)
      (by simpa [one] using first) rfl rfl
    apply HeapCalls.Evaluates.boolean_last
    · simpa [one] using last_test n x env heap getN getX globals
    · cases decide (x = (n : Int) - 1) <;> rfl

/-- The loop proof includes Python's clipping of a nonpositive range stop to zero iterations. -/
private theorem square_loop (remaining : Nat) (n : Nat) (positive : 2 ≤ n)
    (next stop : Int) (equation : (stop - next).toNat = remaining)
    (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getX : env "x" = some (number x))
    (globals : Mutable.Globals) :
    HeapCalls.BlockReturns program env
      [.rangeStepLoop "_" next stop 1 squareBody [], .return (.constant (.bool false))]
      heap (flag (squareSearch n remaining x)) heap globals := by
  induction remaining generalizing next x env with
  | zero =>
      apply HeapCalls.BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  | succ remaining ih =>
      let bound := HeapCalls.setLocal env "_" (number next)
      let squared := x * x % (n : Int)
      let updated := HeapCalls.setLocal bound "x" (number squared)
      have boundN : bound "n" = some (number n) := by simpa [bound, HeapCalls.setLocal] using getN
      have boundX : bound "x" = some (number x) := by simpa [bound, HeapCalls.setLocal] using getX
      have updatedN : updated "n" = some (number n) := by simpa [updated, HeapCalls.setLocal] using boundN
      have updatedX : updated "x" = some (number squared) := by simp [updated, HeapCalls.setLocal]
      have evaluated : HeapCalls.Evaluates program bound squareExpression heap (number squared) heap globals := by
        apply HeapCalls.Evaluates.binary
          (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup boundX) (HeapCalls.Evaluates.lookup boundX) rfl)
          (HeapCalls.Evaluates.lookup boundN)
        simp [squared, Mutable.binary, number, scalarBinary,
          show n ≠ 0 by omega, show 0 < n by omega]
      apply HeapCalls.BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply HeapCalls.BlockReturns.assign evaluated
      by_cases hit : squared = (n : Int) - 1
      · have score : squareSearch n (remaining + 1) x = true := by
          change (if squared = (n : Int) - 1 then true else squareSearch n remaining squared) = true
          rw [if_pos hit]
        rw [score]
        apply HeapCalls.BlockReturns.branch (flag := true)
          (by simpa only [show decide (squared = (n : Int) - 1) = true by simp only [hit, decide_true]]
            using last_test n squared updated heap updatedN updatedX globals) rfl
        exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
      · have score : squareSearch n (remaining + 1) x = squareSearch n remaining squared := by
          change (if squared = (n : Int) - 1 then true else squareSearch n remaining squared) = _
          rw [if_neg hit]
        rw [score]
        apply HeapCalls.BlockReturns.branch (flag := false)
          (by simpa only [show decide (squared = (n : Int) - 1) = false by simp only [hit, decide_false]]
            using last_test n squared updated heap updatedN updatedX globals) rfl
        have rest := ih (next + 1) (by omega) squared updated updatedN updatedX
        simpa [squareBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

private abbrev params := ((findFunctionIn program.body "__formalsw_local_is_prime_check").getD ([], [])).1
private abbrev sourceBody := ((findFunctionIn program.body "__formalsw_local_is_prime_check").getD ([], [])).2

/-- The actual lifted local helper follows all its early returns and preserves every caller heap object. -/
theorem result (a : Int) (n d s : Nat) (positive : 2 ≤ n)
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapCalls.ReturnsValue program "__formalsw_local_is_prime_check"
      [number a, number n, number d, number s] heap (flag (millerCheck a n d s)) heap globals := by
  let env := HeapCalls.functionEnv params sourceBody [number a, number n, number d, number s] globals
  have getA : env "a" = some (number a) := by cases globals <;> rfl
  have getN : env "n" = some (number n) := by cases globals <;> rfl
  have getD : env "d" = some (number d) := by cases globals <;> rfl
  have getS : env "s" = some (number s) := by cases globals <;> rfl
  have remainder : HeapCalls.Evaluates program env
      (.compare (.binOp (.name "a") .mod (.name "n")) [(.eq, .constant (.int 0))])
      heap (flag (decide (a % (n : Int) = 0))) heap globals := by
    have reduced : HeapCalls.Evaluates program env
        (.binOp (.name "a") .mod (.name "n")) heap (number (a % (n : Int))) heap globals := by
      apply HeapCalls.Evaluates.binary (middle := heap) (finalHeap := heap)
        (HeapCalls.Evaluates.lookup getA) (HeapCalls.Evaluates.lookup getN)
      simp [Mutable.binary, number, scalarBinary, show n ≠ 0 by omega, show 0 < n by omega]
    apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
      (av := number (a % (n : Int))) (bv := number 0) reduced HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
  apply HeapCalls.BlockReturns.function rfl rfl
  by_cases divisible : a % (n : Int) = 0
  · rw [millerCheck, if_pos divisible]
    apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [divisible] using remainder) rfl
    exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  · rw [millerCheck, if_neg divisible]
    apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [divisible] using remainder) rfl
    let x := FormalSW.Py.Builtin.Int.powModBinary a d n
    have power : HeapCalls.Evaluates program env
        (.builtinCall .pow [.name "a", .name "d", .name "n"]) heap (number x) heap globals := by
      apply HeapCalls.Evaluates.builtin
        (.cons (HeapCalls.Evaluates.lookup getA)
          (.cons (HeapCalls.Evaluates.lookup getD) (.cons (HeapCalls.Evaluates.lookup getN) .nil)))
      simp [HeapCalls.primitive, HeapCalls.scalarResult, Calls.modularPower, scalarNumber,
        FormalSW.Py.Builtin.Int.powMod_nonnegative a d n (by omega),
        ← FormalSW.Py.Builtin.Int.powModBinary_correct a d n, number, x]
    apply HeapCalls.BlockReturns.assign power
    let updated := HeapCalls.setLocal env "x" (number x)
    have updatedN : updated "n" = some (number n) := by simpa [updated, HeapCalls.setLocal] using getN
    have updatedX : updated "x" = some (number x) := by simp [updated, HeapCalls.setLocal]
    have updatedS : updated "s" = some (number s) := by simpa [updated, HeapCalls.setLocal] using getS
    change HeapCalls.BlockReturns program updated
      [.ifThen initialTest [.return (.constant (.bool true))] [],
       .forIn (.name "_") (.call (.name "range") [.binOp (.name "s") .sub (.constant (.int 1))]) squareBody [],
       .return (.constant (.bool false))] heap
      (flag (if x = 1 ∨ x = (n : Int) - 1 then true else squareSearch n (s - 1) x)) heap globals
    have tested := initial_test n x updated heap updatedN updatedX globals
    by_cases accepted : x = 1 ∨ x = (n : Int) - 1
    · rw [if_pos accepted]
      apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [accepted] using tested) rfl
      exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
    · rw [if_neg accepted]
      apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [accepted] using tested) rfl
      apply HeapCalls.BlockReturns.for_range
        (.cons (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup updatedS) HeapCalls.Evaluates.constant rfl) .nil) rfl
      apply square_loop (s - 1) n positive 0 ((s : Int) - 1) (by omega) x updated heap updatedN updatedX globals

end solvers.«58».miller_check

namespace solvers.«58».decomposition
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin

private abbrev program := Solvers.P58Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def condition : Expr :=
  .compare (.binOp (.name "d") .mod (.constant (.int 2))) [(.eq, .constant (.int 0))]
def body : List Stmt :=
  [.augAssign (.name "s") .add (.constant (.int 1)),
   .augAssign (.name "d") .floorDiv (.constant (.int 2))]
def statement : Stmt := .whileDo condition body []

theorem source_statement :
    (executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2)[5]? = some statement := rfl

private theorem condition_evaluates (d : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getD : env "d" = some (number d)) (globals : Mutable.Globals) :
    HeapCalls.Evaluates program env condition heap (flag (decide (d % 2 = 0))) heap globals := by
  have remainder : HeapCalls.Evaluates program env
      (.binOp (.name "d") .mod (.constant (.int 2))) heap (number ((d % 2 : Nat) : Int)) heap globals := by
    apply HeapCalls.Evaluates.binary (middle := heap) (finalHeap := heap)
      (HeapCalls.Evaluates.lookup getD) HeapCalls.Evaluates.constant
    simp [Mutable.binary, number, scalarBinary]
  apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number ((d % 2 : Nat) : Int)) (bv := number 0) remainder HeapCalls.Evaluates.constant
  by_cases zero : d % 2 = 0
  · simp only [zero]
    rfl
  · have one : d % 2 = 1 := by omega
    simp only [one]
    rfl

/-- The unchanged source loop computes the exact odd-part model while preserving every other local and every heap object. -/
theorem loop (fuel d s : Nat) (positive : 0 < d) (bound : d ≤ fuel)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getD : env "d" = some (number d)) (getS : env "s" = some (number s))
    (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      HeapCalls.BlockContinues program env [statement] heap afterEnv heap globals ∧
      afterEnv "d" = some (number (decomposeAux fuel d s).1) ∧
      afterEnv "s" = some (number (decomposeAux fuel d s).2) ∧
      ∀ name, name ≠ "d" → name ≠ "s" → afterEnv name = env name := by
  induction fuel generalizing d s env with
  | zero => omega
  | succ fuel ih =>
      have evaluated := condition_evaluates d env heap getD globals
      by_cases even : d % 2 = 0
      · rw [decomposeAux, if_pos even]
        have halfPositive : 0 < d / 2 := by omega
        have halfBound : d / 2 ≤ fuel := by omega
        let incremented := HeapCalls.setLocal env "s" (number (s + 1))
        let updated := HeapCalls.setLocal incremented "d" (number (d / 2))
        have incrementedD : incremented "d" = some (number d) := by
          simpa [incremented, HeapCalls.setLocal] using getD
        have updatedD : updated "d" = some (number (d / 2)) := by simp [updated, HeapCalls.setLocal]
        have updatedS : updated "s" = some (number (s + 1)) := by
          simp [updated, incremented, HeapCalls.setLocal]
        obtain ⟨afterEnv, recursively, afterD, afterS, unchanged⟩ :=
          ih (d / 2) (s + 1) halfPositive halfBound updated updatedD updatedS
        refine ⟨afterEnv, ?_, afterD, afterS, ?_⟩
        · apply HeapCalls.BlockContinues.while_loop (flag := true)
            (by simpa [even] using evaluated) rfl
          have increment : HeapCalls.BlockContinues program env
              [.augAssign (.name "s") .add (.constant (.int 1))] heap incremented heap globals := by
            apply HeapCalls.BlockContinues.augAssign getS HeapCalls.Evaluates.constant
            simp [HeapCalls.inplace, Mutable.binary, number, scalarBinary]
          have divide : HeapCalls.BlockContinues program incremented
              [.augAssign (.name "d") .floorDiv (.constant (.int 2))] heap updated heap globals := by
            apply HeapCalls.BlockContinues.augAssign incrementedD HeapCalls.Evaluates.constant
            simp [HeapCalls.inplace, Mutable.binary, number, scalarBinary]
          apply HeapCalls.BlockContinues.append (HeapCalls.BlockContinues.append increment divide)
          simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using recursively
        · intro name notD notS
          calc
            afterEnv name = updated name := unchanged name notD notS
            _ = env name := by simp [updated, incremented, HeapCalls.setLocal, notD, notS]
      · rw [decomposeAux, if_neg even]
        refine ⟨env, ?_, getD, getS, fun _ _ _ => rfl⟩
        apply HeapCalls.BlockContinues.while_loop (flag := false)
          (by simpa [even] using evaluated) rfl
        exact HeapCalls.BlockContinues.nil

/-- Public-fuel specialization used by the outer source is_prime proof. -/
theorem result (d : Nat) (positive : 0 < d) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getD : env "d" = some (number d)) (getS : env "s" = some (number 0))
    (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      HeapCalls.BlockContinues program env [statement] heap afterEnv heap globals ∧
      afterEnv "d" = some (number (decompose d).1) ∧
      afterEnv "s" = some (number (decompose d).2) ∧
      ∀ name, name ≠ "d" → name ≠ "s" → afterEnv name = env name :=
  loop d d 0 positive (Nat.le_refl d) env heap getD getS globals

end solvers.«58».decomposition

namespace solvers.«58».small_prime_scan
open FormalSW.PyAst
variable {globals : Mutable.Globals}
open ProjectEulerSolutions.MillerRabin
private abbrev program := Solvers.P58Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
def numbers (values : List Nat) : List Mutable.Value := values.map (fun (p : Nat) => number p)
def equalityTest : Expr := .compare (.name "n") [(.eq, .name "p")]
def divisionTest : Expr :=
  .compare (.binOp (.name "n") .mod (.name "p")) [(.eq, .constant (.int 0))]
def body : List Stmt :=
  [.ifThen equalityTest [.return (.constant (.bool true))] [],
   .ifThen divisionTest [.return (.constant (.bool false))] []]

def answer (scan : Option Bool) (fallback : Mutable.Value) : Mutable.Value :=
  match scan with | some b => flag b | none => fallback
def finalHeap (scan : Option Bool) (initial fallback : Mutable.Heap) : Mutable.Heap :=
  match scan with | some _ => initial | none => fallback

/-- Exact live-sequence traversal, including both early returns and the unchanged caller heap. -/
theorem loop (n : Int) (values remaining : List Nat) (position : Nat)
    (suffix : values.drop position = remaining) (positive : ∀ p ∈ remaining, 0 < p)
    (source : SequenceSource) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (sequence : Mutable.sequence (HeapCalls.sequenceValue source) heap = some (.ok (numbers values), heap))
    (getN : env "n" = some (number n))
    (following : List Stmt) (fallback : Mutable.Value) (fallbackHeap : Mutable.Heap)
    (continued : smallScan n remaining = none →
      ∀ afterEnv, (∀ name, name ≠ "p" → afterEnv name = env name) →
      HeapCalls.BlockReturns program afterEnv following heap fallback fallbackHeap globals) :
    HeapCalls.BlockReturns program env
      (.sequenceStepLoop "p" source position body [] :: following) heap
      (answer (smallScan n remaining) fallback) (finalHeap (smallScan n remaining) heap fallbackHeap) globals := by
  induction remaining generalizing position env with
  | nil =>
      have exhausted : values[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      simp only [smallScan, answer, finalHeap]
      apply HeapCalls.BlockReturns.sequence_done sequence
        (by simp [numbers, exhausted])
      exact continued rfl env (fun _ _ => rfl)
  | cons p remaining ih =>
      have present : values[position]? = some p := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      have nextSuffix : values.drop (position + 1) = remaining := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      have pPositive := positive p (by simp)
      let bound := HeapCalls.setLocal env "p" (number p)
      have boundN : bound "n" = some (number n) := by simpa [bound, HeapCalls.setLocal] using getN
      have boundP : bound "p" = some (number p) := by simp [bound, HeapCalls.setLocal]
      have equalTest : HeapCalls.Evaluates program bound equalityTest heap
          (flag (decide (n = (p : Int)))) heap globals := by
        apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
          (HeapCalls.Evaluates.lookup boundN) (HeapCalls.Evaluates.lookup boundP)
        simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
      apply HeapCalls.BlockReturns.sequence_step (value := number p) sequence (by simp [numbers, present])
      by_cases equal : n = (p : Int)
      · simp only [smallScan, if_pos equal, answer, finalHeap]
        apply HeapCalls.BlockReturns.branch (flag := true)
          (by simpa only [equal, decide_true] using equalTest) rfl
        exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
      · simp only [smallScan, if_neg equal]
        apply HeapCalls.BlockReturns.branch (flag := false)
          (by simpa only [equal, decide_false] using equalTest) rfl
        have reduced : HeapCalls.Evaluates program bound
            (.binOp (.name "n") .mod (.name "p")) heap (number (n % (p : Int))) heap globals := by
          apply HeapCalls.Evaluates.binary (middle := heap) (finalHeap := heap)
            (HeapCalls.Evaluates.lookup boundN) (HeapCalls.Evaluates.lookup boundP)
          simp [Mutable.binary, number, scalarBinary, pPositive, Nat.ne_of_gt pPositive]
        have dividesTest : HeapCalls.Evaluates program bound divisionTest heap
            (flag (decide (n % (p : Int) = 0))) heap globals := by
          apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
            (av := number (n % (p : Int))) (bv := number 0) reduced HeapCalls.Evaluates.constant
          simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
        by_cases divides : n % (p : Int) = 0
        · simp only [if_pos divides, answer, finalHeap]
          apply HeapCalls.BlockReturns.branch (flag := true)
            (by simpa only [divides, decide_true] using dividesTest) rfl
          exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
        · simp only [if_neg divides]
          apply HeapCalls.BlockReturns.branch (flag := false)
            (by simpa only [divides, decide_false] using dividesTest) rfl
          have rest := ih (position + 1) nextSuffix
            (fun q member => positive q (by simp [member])) bound boundN
            (fun tailMiss afterEnv unchanged => continued
              (by simpa only [smallScan, if_neg equal, if_neg divides] using tailMiss)
              afterEnv (fun name notP =>
                (unchanged name notP).trans (by simp [bound, HeapCalls.setLocal, notP])))
          simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

/-- The source for statement observes the existing list object and evaluates its iterable once. -/
theorem result (n : Int) (values : List Nat) (positive : ∀ p ∈ values, 0 < p)
    (address : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list (numbers values)))
    (getValues : env "small_primes" = some (.ref address)) (getN : env "n" = some (number n))
    (following : List Stmt) (fallback : Mutable.Value) (fallbackHeap : Mutable.Heap)
    (continued : smallScan n values = none →
      ∀ afterEnv, (∀ name, name ≠ "p" → afterEnv name = env name) →
      HeapCalls.BlockReturns program afterEnv following heap fallback fallbackHeap globals) :
    HeapCalls.BlockReturns program env
      (.forIn (.name "p") (.name "small_primes") body [] :: following) heap
      (answer (smallScan n values) fallback) (finalHeap (smallScan n values) heap fallbackHeap) globals := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (numbers values), heap) := by
    simp [Mutable.sequence, found]
  apply HeapCalls.BlockReturns.for_sequence (by intros; simp)
    (HeapCalls.Evaluates.lookup getValues) sequence
  exact loop n values values 0 (by simp) positive (.ref address) env heap sequence getN
    following fallback fallbackHeap continued

end solvers.«58».small_prime_scan

namespace solvers.«58».small_prime_scan
open FormalSW.PyAst
variable {globals : Mutable.Globals}
open ProjectEulerSolutions.MillerRabin
private abbrev scanModule := Solvers.P58Ast.module
private abbrev scanNumber (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev scanFlag (value : Bool) : Mutable.Value := .scalar (.bool value)

/-- A none result proves that every actual early-return branch is skipped. -/
theorem continue_loop (n : Int) (values remaining : List Nat) (position : Nat)
    (suffix : values.drop position = remaining) (positive : ∀ p ∈ remaining, 0 < p)
    (miss : smallScan n remaining = none) (source : SequenceSource)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (sequence : Mutable.sequence (HeapCalls.sequenceValue source) heap = some (.ok (numbers values), heap))
    (getN : env "n" = some (scanNumber n)) (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      HeapCalls.BlockContinues scanModule env [.sequenceStepLoop "p" source position body []]
        heap afterEnv heap globals ∧
      ∀ name, name ≠ "p" → afterEnv name = env name := by
  induction remaining generalizing position env with
  | nil =>
      have exhausted : values[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      refine ⟨env, ?_, fun _ _ => rfl⟩
      apply HeapCalls.BlockContinues.sequence_done sequence (by simp [numbers, exhausted])
      exact HeapCalls.BlockContinues.nil
  | cons p remaining ih =>
      have present : values[position]? = some p := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      have nextSuffix : values.drop (position + 1) = remaining := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      have pPositive := positive p (by simp)
      have unequal : n ≠ (p : Int) := by
        intro equal
        simp [smallScan, equal] at miss
      have nondividing : n % (p : Int) ≠ 0 := by
        intro divides
        simp [smallScan, unequal, divides] at miss
      have missTail : smallScan n remaining = none := by
        simpa only [smallScan, if_neg unequal, if_neg nondividing] using miss
      let bound := HeapCalls.setLocal env "p" (scanNumber p)
      have boundN : bound "n" = some (scanNumber n) := by simpa [bound, HeapCalls.setLocal] using getN
      have boundP : bound "p" = some (scanNumber p) := by simp [bound, HeapCalls.setLocal]
      have equalTest : HeapCalls.Evaluates scanModule bound equalityTest heap (scanFlag false) heap globals := by
        apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
          (HeapCalls.Evaluates.lookup boundN) (HeapCalls.Evaluates.lookup boundP)
        simp [HeapCalls.compare, scanFlag, scalarCompare, evalCmpOp, unequal]
      have reduced : HeapCalls.Evaluates scanModule bound
          (.binOp (.name "n") .mod (.name "p")) heap (scanNumber (n % (p : Int))) heap globals := by
        apply HeapCalls.Evaluates.binary (middle := heap) (finalHeap := heap)
          (HeapCalls.Evaluates.lookup boundN) (HeapCalls.Evaluates.lookup boundP)
        simp [Mutable.binary, scanNumber, scalarBinary, pPositive, Nat.ne_of_gt pPositive]
      have dividesTest : HeapCalls.Evaluates scanModule bound divisionTest heap (scanFlag false) heap globals := by
        apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
          (av := scanNumber (n % (p : Int))) (bv := scanNumber 0) reduced HeapCalls.Evaluates.constant
        simp [HeapCalls.compare, scanFlag, scalarCompare, evalCmpOp, nondividing]
      have skipsEquality : HeapCalls.BlockContinues scanModule bound
          [.ifThen equalityTest [.return (.constant (.bool true))] []] heap bound heap globals :=
        HeapCalls.BlockContinues.branch equalTest rfl HeapCalls.BlockContinues.nil
      have skipsDivision : HeapCalls.BlockContinues scanModule bound
          [.ifThen divisionTest [.return (.constant (.bool false))] []] heap bound heap globals :=
        HeapCalls.BlockContinues.branch dividesTest rfl HeapCalls.BlockContinues.nil
      obtain ⟨afterEnv, recursively, unchanged⟩ := ih (position + 1) nextSuffix
        (fun q member => positive q (by simp [member])) missTail bound boundN
      refine ⟨afterEnv, ?_, ?_⟩
      · apply HeapCalls.BlockContinues.sequence_step (value := scanNumber p) sequence (by simp [numbers, present])
        apply HeapCalls.BlockContinues.append
          (HeapCalls.BlockContinues.append skipsEquality skipsDivision)
        simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using recursively
      · intro name notP
        exact (unchanged name notP).trans (by simp [bound, HeapCalls.setLocal, notP])

/-- The entire source small-prime for statement continues exactly when the pure scan is none. -/
theorem continue_result (n : Int) (values : List Nat) (positive : ∀ p ∈ values, 0 < p)
    (miss : smallScan n values = none) (address : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (found : heap[address]? = some (.list (numbers values)))
    (getValues : env "small_primes" = some (.ref address)) (getN : env "n" = some (scanNumber n))
    (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      HeapCalls.BlockContinues scanModule env [.forIn (.name "p") (.name "small_primes") body []]
        heap afterEnv heap globals ∧
      ∀ name, name ≠ "p" → afterEnv name = env name := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (numbers values), heap) := by
    simp [Mutable.sequence, found]
  obtain ⟨afterEnv, scanned, unchanged⟩ := continue_loop n values values 0 (by simp) positive miss
    (.ref address) env heap sequence getN globals
  refine ⟨afterEnv, ?_, unchanged⟩
  exact HeapCalls.BlockContinues.for_sequence (by intros; simp)
    (HeapCalls.Evaluates.lookup getValues) sequence scanned

end solvers.«58».small_prime_scan

namespace solvers.«58».bases_loop
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin
open ProjectEulerSolutions
private abbrev program := Solvers.P58Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
def numbers (values : List Int) : List Mutable.Value := values.map number

def callExpression : Expr := .call (.name "__formalsw_local_is_prime_check")
  [.name "a", .name "n", .name "d", .name "s"]
def condition : Expr := .unaryOp .not callExpression
def body : List Stmt := [.ifThen condition [.return (.constant (.bool false))] []]

/-- Every iteration calls the real lifted helper with its captured n,d,s values. -/
theorem loop (n d s : Nat) (positive : 2 ≤ n) (values remaining : List Int) (position : Nat)
    (suffix : values.drop position = remaining) (source : SequenceSource)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (sequence : Mutable.sequence (HeapCalls.sequenceValue source) heap = some (.ok (numbers values), heap))
    (getN : env "n" = some (number n)) (getD : env "d" = some (number d))
    (getS : env "s" = some (number s)) (globals : Mutable.Globals := []) :
    HeapCalls.BlockReturns program env
      [.sequenceStepLoop "a" source position body [], .return (.constant (.bool true))]
      heap (flag (remaining.all (fun a => millerCheck a n d s))) heap globals := by
  induction remaining generalizing position env with
  | nil =>
      have exhausted : values[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      apply HeapCalls.BlockReturns.sequence_done sequence (by simp [numbers, exhausted])
      exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  | cons a remaining ih =>
      have present : values[position]? = some a := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      have nextSuffix : values.drop (position + 1) = remaining := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := HeapCalls.setLocal env "a" (number a)
      have boundA : bound "a" = some (number a) := by simp [bound, HeapCalls.setLocal]
      have boundN : bound "n" = some (number n) := by simpa [bound, HeapCalls.setLocal] using getN
      have boundD : bound "d" = some (number d) := by simpa [bound, HeapCalls.setLocal] using getD
      have boundS : bound "s" = some (number s) := by simpa [bound, HeapCalls.setLocal] using getS
      have called : HeapCalls.Evaluates program bound callExpression heap (flag (millerCheck a n d s)) heap globals := by
        apply HeapCalls.Evaluates.call
          (.cons (HeapCalls.Evaluates.lookup boundA)
            (.cons (HeapCalls.Evaluates.lookup boundN)
              (.cons (HeapCalls.Evaluates.lookup boundD) (.cons (HeapCalls.Evaluates.lookup boundS) .nil))))
        exact miller_check.result a n d s positive heap globals
      have tested : HeapCalls.Evaluates program bound condition heap (flag (!millerCheck a n d s)) heap globals := by
        apply HeapCalls.Evaluates.unary called
        cases millerCheck a n d s <;> rfl
      apply HeapCalls.BlockReturns.sequence_step (value := number a) sequence (by simp [numbers, present])
      cases passed : millerCheck a n d s with
      | false =>
          simp only [List.all_cons, passed, Bool.false_and]
          apply HeapCalls.BlockReturns.branch (flag := true) (by simpa only [passed, Bool.not_false] using tested) rfl
          exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
      | true =>
          simp only [List.all_cons, passed, Bool.true_and]
          apply HeapCalls.BlockReturns.branch (flag := false) (by simpa only [passed, Bool.not_true] using tested) rfl
          have rest := ih (position + 1) nextSuffix bound boundN boundD boundS
          simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

/-- Pool identity is extracted from the actual is_prime for statement. -/
def expression : Expr :=
  match (executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2).drop 6 with
  | .forIn _ iterable _ _ :: _ => iterable
  | _ => .constant .none
def identity : String := (HeapMemoCalls.pooledIdentity expression).getD ""
def key : HeapMemoCalls.Key := HeapMemoCalls.literalKey identity

theorem expression_eq : expression = HeapMemoCalls.pooledTuple identity
    [.constant (.int 2), .constant (.int 325), .constant (.int 9375), .constant (.int 28178),
     .constant (.int 450775), .constant (.int 9780504), .constant (.int 1795265022)] := rfl

def Found (heap : Mutable.Heap) (value : Mutable.Value) : Prop :=
  ∃ address, value = .ref address ∧ heap[address]? = some (.tuple (numbers bases))
def spec (heap : Mutable.Heap) (query : HeapMemoCalls.Key) (value : Mutable.Value) : Prop :=
  query = key ∧ Found heap value

theorem found_append (heap extra : Mutable.Heap) (value : Mutable.Value) (found : Found heap value) :
    Found (heap ++ extra) value := by
  obtain ⟨address, rfl, found⟩ := found
  exact ⟨address, rfl, HeapCollection.prefix_found heap extra address _ found⟩

theorem spec_preserves_append (heap extra : Mutable.Heap) :
    HeapMemoCalls.Cache.Preserves spec heap (heap ++ extra) := by
  intro cache valid query value cached
  obtain ⟨same, found⟩ := valid query value cached
  exact ⟨same, found_append heap extra value found⟩

def PoolPost (heap : Mutable.Heap) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ extra, after = heap ++ extra ∧ Found after value

/-- Both a real pool miss and an existing-reference pool hit supply the identical live tuple. -/
theorem evaluate (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapMemoCalls.EvaluatesRel program spec env expression heap (PoolPost heap) globals := by
  rw [expression_eq]
  apply HeapMemoCalls.EvaluatesRel.pooled_tuple
  · have allocated := HeapMemoCalls.Evaluates.tuple (module := program) (spec := spec)
      (env := env) (heap := heap) (globals := globals)
      (.cons (HeapMemoCalls.Evaluates.constant (value := .int 2))
        (.cons (HeapMemoCalls.Evaluates.constant (value := .int 325))
          (.cons (HeapMemoCalls.Evaluates.constant (value := .int 9375))
            (.cons (HeapMemoCalls.Evaluates.constant (value := .int 28178))
              (.cons (HeapMemoCalls.Evaluates.constant (value := .int 450775))
                (.cons (HeapMemoCalls.Evaluates.constant (value := .int 9780504))
                  (.cons (HeapMemoCalls.Evaluates.constant (value := .int 1795265022)) .nil)))))))
      (spec_preserves_append heap [.tuple (numbers bases)]) (by simp)
    apply allocated.rel.consequence
    rintro value after ⟨valueEq, heapEq⟩
    subst value
    subst after
    have found : Found (heap ++ [.tuple (numbers bases)]) (.ref heap.length) :=
      ⟨heap.length, rfl, by simp⟩
    exact ⟨⟨[.tuple (numbers bases)], rfl, found⟩, rfl, found⟩
  · intro value cached
    exact ⟨[], by simp, cached.2⟩

end solvers.«58».bases_loop

namespace solvers.«58».bases_loop
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin
private abbrev module58 := Solvers.P58Ast.module

private theorem cache_disjoint : HeapMemoCalls.CacheDisjoint module58 module58 := by
  intro name cached
  change name ∈ ([] : List String) at cached
  simp at cached

/-- The actual pooled for statement evaluates its iterable once and checks all seven bases in source order. -/
theorem result (n d s : Nat) (positive : 2 ≤ n) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (.scalar (.int n)))
    (getD : env "d" = some (.scalar (.int d)))
    (getS : env "s" = some (.scalar (.int s))) (globals : Mutable.Globals := []) :
    HeapMemoCalls.BlockReturnsRel module58 spec env
      [.forIn (.name "a") expression body [], .return (.constant (.bool true))] heap
      (fun value after => value = .scalar (.bool (bases.all (fun a => millerCheck a n d s))) ∧
        ∃ extra, after = heap ++ extra) globals := by
  intro anchors stack
  apply HeapMemoCalls.RunsRel.next
    (current := ⟨.block [.forIn (.name "a") expression body [], .return (.constant (.bool true))] env, stack, heap⟩)
    (following := ⟨.expr expression env,
      .sequence "a" body [] [.return (.constant (.bool true))] env :: stack, heap⟩)
    (by trivial) (by rw [expression_eq]; rfl)
  apply (evaluate env heap globals anchors
    (.sequence "a" body [] [.return (.constant (.bool true))] env :: stack)).bind
  rintro final ⟨value, after, rfl, extra, shape, address, rfl, found⟩
  have sequence : Mutable.sequence (.ref address) after = some (.ok (numbers bases), after) := by
    simp [Mutable.sequence, found]
  apply HeapMemoCalls.RunsRel.next
    (current := ⟨.value (.ref address),
      .sequence "a" body [] [.return (.constant (.bool true))] env :: stack, after⟩)
    (following := ⟨.block [.sequenceStepLoop "a" (.ref address) 0 body [],
      .return (.constant (.bool true))] env, stack, after⟩)
    (by trivial) (by simp [HeapCalls.step, HeapCalls.run, sequence, HeapCalls.sequenceSource])
  have plain := loop n d s positive bases bases 0 (by simp) (.ref address) env after
    sequence getN getD getS globals
  intro callers cache valid
  refine ⟨⟨.value (.scalar (.bool (bases.all (fun a => millerCheck a n d s)))), stack, after⟩,
    cache, ?_, valid, ?_⟩
  · exact ⟨_, _, rfl, rfl, extra, shape⟩
  · exact HeapMemoCalls.Reaches.from_heap (HeapCalls.Extends.refl _) cache_disjoint (plain anchors stack)

end solvers.«58».bases_loop

namespace FormalSW.PyAst.HeapMemoCalls

/-- An ordinary allocating prefix may precede a cache-aware suffix when its actual heap extension preserves cache entries. -/
theorem BlockReturnsRel.after_heap_prefix_frame
    (disjoint : CacheDisjoint module module)
    (beforeRun : HeapCalls.BlockContinues module env body heap afterEnv afterHeap globals)
    (preserved : Cache.Preserves spec heap afterHeap)
    (rest : BlockReturnsRel module spec afterEnv following afterHeap post globals) :
    BlockReturnsRel module spec env (body ++ following) heap post globals := by
  intro anchors stack callers cache valid
  obtain ⟨final, updated, holds, validFinal, suffix⟩ :=
    rest anchors stack callers cache (preserved cache valid)
  refine ⟨final, updated, holds, validFinal, ?_⟩
  exact (Reaches.from_heap (HeapCalls.Extends.refl _) disjoint (beforeRun following anchors stack)).trans suffix

end FormalSW.PyAst.HeapMemoCalls

namespace solvers.«58».is_prime_prefix
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin
private abbrev module58 := Solvers.P58Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def condition : Expr := .compare (.name "n") [(.lt, .constant (.int 2))]
def smallExpression : Expr := .list (smallPrimes.map (fun (p : Nat) => .constant (.int p)))
def smallStatement : Stmt := .annAssign (.name "small_primes")
  (.subscript (.name "list") (.name "int")) (some smallExpression)
def leadingCode : List Stmt :=
  [.ifThen condition [.return (.constant (.bool false))] [], smallStatement,
   .forIn (.name "p") (.name "small_primes") small_prime_scan.body [],
   .assign [.name "d"] (.binOp (.name "n") .sub (.constant (.int 1))),
   .assign [.name "s"] (.constant (.int 0)), decomposition.statement]

theorem prefix_eq :
    (executableBody ((findFunctionIn module58.body "is_prime").getD ([], [])).2).take 6 = leadingCode := rfl

private theorem literal_list (values : List Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) :
    HeapCalls.Evaluates module58 env (.list (values.map (fun (p : Nat) => .constant (.int p)))) heap
      (.ref heap.length) (heap ++ [.list (small_prime_scan.numbers values)]) globals := by
  apply HeapCalls.Evaluates.list
  induction values with
  | nil => exact .nil
  | cons value values ih => exact .cons HeapCalls.Evaluates.constant ih

/-- The complete source prefix reaches the pooled bases with the exact odd part and every caller heap object retained. -/
theorem result (n : Nat) (positive : 2 ≤ n) (miss : smallScan n smallPrimes = none)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getN : env "n" = some (number n))
    (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      HeapCalls.BlockContinues module58 env leadingCode heap afterEnv
        (heap ++ [.list (small_prime_scan.numbers smallPrimes)]) globals ∧
      afterEnv "n" = some (number n) ∧
      afterEnv "d" = some (number (decompose (n - 1)).1) ∧
      afterEnv "s" = some (number (decompose (n - 1)).2) := by
  let privateHeap := heap ++ [.list (small_prime_scan.numbers smallPrimes)]
  let allocated := HeapCalls.setLocal env "small_primes" (.ref heap.length)
  have small : HeapCalls.Evaluates module58 env condition heap (flag false) heap globals := by
    apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
      (HeapCalls.Evaluates.lookup getN) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp, show ¬(n : Int) < 2 by omega]
  have skipsSmall : HeapCalls.BlockContinues module58 env
      [.ifThen condition [.return (.constant (.bool false))] []] heap env heap globals :=
    HeapCalls.BlockContinues.branch small rfl HeapCalls.BlockContinues.nil
  have allocates : HeapCalls.BlockContinues module58 env [smallStatement] heap allocated privateHeap globals :=
    HeapCalls.BlockContinues.annAssign (literal_list smallPrimes env heap globals)
  have allocatedN : allocated "n" = some (number n) := by
    simpa [allocated, HeapCalls.setLocal] using getN
  have allocatedValues : allocated "small_primes" = some (.ref heap.length) := by
    simp [allocated, HeapCalls.setLocal]
  have privateFound : privateHeap[heap.length]? = some (.list (small_prime_scan.numbers smallPrimes)) := by
    simp [privateHeap]
  obtain ⟨scanned, scans, unchanged⟩ := small_prime_scan.continue_result n smallPrimes
    (by decide) miss heap.length allocated privateHeap privateFound allocatedValues allocatedN globals
  have scannedN : scanned "n" = some (number n) := by rw [unchanged "n" (by decide)]; exact allocatedN
  let initializedD := HeapCalls.setLocal scanned "d" (number (n - 1 : Nat))
  let initialized := HeapCalls.setLocal initializedD "s" (number 0)
  have initializeD : HeapCalls.BlockContinues module58 scanned
      [.assign [.name "d"] (.binOp (.name "n") .sub (.constant (.int 1)))]
      privateHeap initializedD privateHeap globals := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (middle := privateHeap) (finalHeap := privateHeap)
      (HeapCalls.Evaluates.lookup scannedN) HeapCalls.Evaluates.constant
    simp [Mutable.binary, number, scalarBinary, Int.natCast_sub (show 1 ≤ n by omega)]
  have initializeS : HeapCalls.BlockContinues module58 initializedD
      [.assign [.name "s"] (.constant (.int 0))] privateHeap initialized privateHeap globals :=
    HeapCalls.BlockContinues.assign HeapCalls.Evaluates.constant
  have initializedDValue : initialized "d" = some (number (n - 1 : Nat)) := by
    simp [initialized, initializedD, HeapCalls.setLocal]
  have initializedS : initialized "s" = some (number 0) := by simp [initialized, HeapCalls.setLocal]
  have initializedN : initialized "n" = some (number n) := by
    simpa [initialized, initializedD, HeapCalls.setLocal] using scannedN
  obtain ⟨afterEnv, decomposes, afterD, afterS, otherLocals⟩ :=
    decomposition.result (n - 1) (by omega) initialized privateHeap initializedDValue initializedS globals
  refine ⟨afterEnv, ?_, ?_, afterD, afterS⟩
  · have first := HeapCalls.BlockContinues.append skipsSmall allocates
    have second := HeapCalls.BlockContinues.append first scans
    have third := HeapCalls.BlockContinues.append second initializeD
    have fourth := HeapCalls.BlockContinues.append third initializeS
    simpa only [leadingCode, List.append_assoc, List.nil_append] using
      HeapCalls.BlockContinues.append fourth decomposes
  · rw [otherLocals "n" (by decide) (by decide)]
    exact initializedN

end solvers.«58».is_prime_prefix

namespace solvers.«58».is_prime
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin
private abbrev module58 := Solvers.P58Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev params := ((findFunctionIn module58.body "is_prime").getD ([], [])).1
private abbrev sourceBody := ((findFunctionIn module58.body "is_prime").getD ([], [])).2
private abbrev code := executableBody sourceBody
private abbrev entry (n : Int) (globals : Mutable.Globals) : HeapCalls.Env :=
  HeapCalls.functionEnv params sourceBody [number n] globals

def Post (heap : Mutable.Heap) (n : Nat) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  value = flag (isPrimeModel n) ∧ ∃ extra, after = heap ++ extra

private theorem cache_disjoint : HeapMemoCalls.CacheDisjoint module58 module58 := by
  intro name cached
  change name ∈ ([] : List String) at cached
  simp at cached

private theorem small_test (n : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (globals : Mutable.Globals) :
    HeapCalls.Evaluates module58 env is_prime_prefix.condition heap
      (flag (decide (n < 2))) heap globals := by
  apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
    (HeapCalls.Evaluates.lookup getN) HeapCalls.Evaluates.constant
  simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]

private theorem literal_list (values : List Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) :
    HeapCalls.Evaluates module58 env (.list (values.map (fun (p : Nat) => .constant (.int p)))) heap
      (.ref heap.length) (heap ++ [.list (small_prime_scan.numbers values)]) globals := by
  apply HeapCalls.Evaluates.list
  induction values with
  | nil => exact .nil
  | cons value values ih => exact .cons HeapCalls.Evaluates.constant ih

/-- The exact source literal allocation retains every existing caller object. -/
theorem rawAllocSmallPrimes (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapCalls.Evaluates module58 env is_prime_prefix.smallExpression heap (.ref heap.length)
      (heap ++ [.list (small_prime_scan.numbers smallPrimes)]) globals :=
  literal_list smallPrimes env heap globals

/-- The real signed early-return branch covers negative integers and both 0 and 1 without allocating. -/
theorem small_result (n : Int) (small : n < 2) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapMemoCalls.ReturnsValue module58 bases_loop.spec "is_prime" [number n] heap (flag false) heap globals := by
  have ordinary : HeapCalls.ReturnsValue module58 "is_prime" [number n] heap (flag false) heap globals := by
    have getN : entry n globals "n" = some (number n) := by cases globals <;> rfl
    apply HeapCalls.BlockReturns.function rfl rfl
    apply HeapCalls.BlockReturns.branch (flag := true)
      (by simpa only [small, decide_true] using small_test n (entry n globals) heap getN globals) rfl
    exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  exact HeapMemoCalls.ReturnsValue.from_heap (HeapCalls.Extends.refl _) cache_disjoint ordinary
    HeapMemoCalls.Cache.Preserves.refl

private theorem small_scan_result (n : Nat) (positive : 2 ≤ n) (b : Bool)
    (known : smallScan n smallPrimes = some b) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapCalls.ReturnsValue module58 "is_prime" [number n] heap (flag b)
      (heap ++ [.list (small_prime_scan.numbers smallPrimes)]) globals := by
  let env := entry n globals
  let privateHeap := heap ++ [.list (small_prime_scan.numbers smallPrimes)]
  let allocated := HeapCalls.setLocal env "small_primes" (.ref heap.length)
  have getN : env "n" = some (number n) := by cases globals <;> rfl
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.branch (flag := false)
    (by simpa only [show ¬ (n : Int) < 2 by omega, decide_false]
      using small_test n env heap getN globals) rfl
  have allocates : HeapCalls.BlockContinues module58 env [is_prime_prefix.smallStatement]
      heap allocated privateHeap globals :=
    HeapCalls.BlockContinues.annAssign (rawAllocSmallPrimes env heap globals)
  apply HeapCalls.BlockContinues.returns allocates
  have allocatedN : allocated "n" = some (number n) := by
    simpa [allocated, HeapCalls.setLocal] using getN
  have allocatedValues : allocated "small_primes" = some (.ref heap.length) := by
    simp [allocated, HeapCalls.setLocal]
  have found : privateHeap[heap.length]? = some (.list (small_prime_scan.numbers smallPrimes)) := by
    simp [privateHeap]
  have scanned := small_prime_scan.result (globals := globals) n smallPrimes (by decide) heap.length allocated privateHeap
    found allocatedValues allocatedN (code.drop 3) (.scalar .none) privateHeap
    (fun impossible _ _ => by rw [known] at impossible; contradiction)
  simpa only [known, small_prime_scan.answer, small_prime_scan.finalHeap] using scanned

/-- The actual source primality routine implements its complete Miller–Rabin algorithm for every natural number.
This theorem makes no unrestricted claim that Miller–Rabin is equivalent to mathematical primality. -/
theorem result (n : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapMemoCalls.ReturnsRel module58 bases_loop.spec "is_prime" [number n] heap (Post heap n) globals := by
  by_cases small : n < 2
  · apply (small_result n (by omega) heap globals).rel.consequence
    rintro value after ⟨valueEq, heapEq⟩
    subst value
    subst after
    refine ⟨?_, [], by simp⟩
    simp [isPrimeModel, small]
  · have positive : 2 ≤ n := by omega
    cases known : smallScan n smallPrimes with
    | some b =>
        have ordinary := small_scan_result n positive b known heap globals
        have lifted := HeapMemoCalls.ReturnsValue.from_heap (HeapCalls.Extends.refl _) cache_disjoint ordinary
          (bases_loop.spec_preserves_append heap [.list (small_prime_scan.numbers smallPrimes)])
        apply lifted.rel.consequence
        rintro value after ⟨valueEq, heapEq⟩
        subst value
        subst after
        refine ⟨?_, [.list (small_prime_scan.numbers smallPrimes)], rfl⟩
        simp only [isPrimeModel, if_neg small, known]
    | none =>
        let env := entry n globals
        let privateHeap := heap ++ [.list (small_prime_scan.numbers smallPrimes)]
        have getN : env "n" = some (number n) := by cases globals <;> rfl
        obtain ⟨afterEnv, prefixRun, afterN, afterD, afterS⟩ :=
          is_prime_prefix.result n positive known env heap getN globals
        have suffixRun := bases_loop.result n (decompose (n - 1)).1 (decompose (n - 1)).2 positive
          afterEnv privateHeap afterN afterD afterS globals
        apply HeapMemoCalls.BlockReturnsRel.function (by decide) rfl rfl
        change HeapMemoCalls.BlockReturnsRel module58 bases_loop.spec env
          (is_prime_prefix.leadingCode ++ [.forIn (.name "a") bases_loop.expression bases_loop.body [],
            .return (.constant (.bool true))]) heap (Post heap n) globals
        apply HeapMemoCalls.BlockReturnsRel.after_heap_prefix_frame cache_disjoint prefixRun
          (bases_loop.spec_preserves_append heap [.list (small_prime_scan.numbers smallPrimes)])
        apply suffixRun.consequence
        rintro value after ⟨valueEq, extra, shape⟩
        refine ⟨?_, [.list (small_prime_scan.numbers smallPrimes)] ++ extra, ?_⟩
        · simpa only [isPrimeModel, if_neg small, known] using valueEq
        · simpa only [privateHeap, List.append_assoc] using shape

end solvers.«58».is_prime

namespace solvers.«58».solve
open FormalSW.PyAst
open ProjectEulerSolutions
private abbrev module58 := Solvers.P58Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev natural (value : Nat) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def encode (values : List Nat) : List Mutable.Value := values.map (fun (n : Nat) => natural n)
def modelCount (values : List Nat) : Nat := (values.filter MillerRabin.isPrimeModel).length

def ScalarPost (heap : Mutable.Heap) (expected : Int) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  value = number expected ∧ ∃ extra, after = heap ++ extra

theorem ScalarPost.frame (shape : middle = heap ++ extra) (holds : ScalarPost middle expected value after) :
    ScalarPost heap expected value after := by
  obtain ⟨equal, tail, finalShape⟩ := holds
  exact ⟨equal, extra ++ tail, by rw [finalShape, shape, List.append_assoc]⟩

def primeExpression : Expr := .call (.name "is_prime") [.name "x"]
def filteredBody : List Stmt :=
  HeapCalls.comprehensionFilters [primeExpression] (HeapCalls.sumBody (.constant (.int 1)) [])
def sumExpression : Expr := .sumGenerator (.constant (.int 1))
  [("x", .name "corners", [primeExpression])] (.constant (.int 0))

private theorem tuple_sequence (heap : Mutable.Heap) (address : Nat) (values : List Nat)
    (found : heap[address]? = some (.tuple (encode values))) :
    Mutable.sequence (.ref address) heap = some (.ok (encode values), heap) := by
  simp [Mutable.sequence, found]

def FlagPost (heap : Mutable.Heap) (expected : Bool) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  value = flag expected ∧ ∃ extra, after = heap ++ extra

/-- Each lazy filter invokes the real source primality function, retaining all private allocations and pool hits. -/
private theorem prime_evaluates (n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getX : env "x" = some (natural n)) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel module58 bases_loop.spec env primeExpression heap
      (FlagPost heap (MillerRabin.isPrimeModel n)) globals := by
  apply HeapMemoCalls.EvaluatesRel.call
    (HeapMemoCalls.ArgumentsRel.fixed (.cons (HeapMemoCalls.Evaluates.lookup getX) .nil))
  rintro values middle ⟨valueEq, heapEq⟩
  subst values
  subst middle
  exact is_prime.result n heap globals

/-- The live tuple is reacquired after each effectful filter, and the hidden accumulator changes only on accepted values. -/
theorem generator_loop (values remaining : List Nat) (position : Nat)
    (suffix : values.drop position = remaining) (address : Nat) (heap : Mutable.Heap)
    (found : heap[address]? = some (.tuple (encode values))) (env : HeapCalls.Env) (initial : Nat)
    (getAccumulator : env HeapCalls.sumAccumulator = some (natural initial)) (globals : Mutable.Globals := []) :
    HeapMemoCalls.BlockReturnsRel module58 bases_loop.spec env
      [.sequenceStepLoop "x" (.ref address) position filteredBody [], .return (.name HeapCalls.sumAccumulator)]
      heap (ScalarPost heap ((initial + modelCount remaining : Nat) : Int)) globals := by
  induction remaining generalizing position heap env initial with
  | nil =>
      have exhausted : values[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      simp only [modelCount, List.filter_nil, List.length_nil, Nat.add_zero]
      apply HeapMemoCalls.BlockReturnsRel.sequence_done (source := .ref address) (items := encode values)
        (by simpa only [HeapCalls.sequenceValue] using tuple_sequence heap address values found)
        (by simp [encode, exhausted])
      exact HeapMemoCalls.BlockReturnsRel.return_value
        (HeapMemoCalls.EvaluatesRel.lookup getAccumulator ⟨rfl, [], by simp⟩)
  | cons n remaining ih =>
      have present : values[position]? = some n := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      have nextSuffix : values.drop (position + 1) = remaining := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := HeapCalls.setLocal env "x" (natural n)
      have boundX : bound "x" = some (natural n) := by simp [bound, HeapCalls.setLocal]
      have boundAccumulator : bound HeapCalls.sumAccumulator = some (natural initial) := by
        simpa [bound, HeapCalls.setLocal, HeapCalls.sumAccumulator] using getAccumulator
      have tested := prime_evaluates n bound heap boundX globals
      apply HeapMemoCalls.BlockReturnsRel.sequence_step (value := natural n)
        (source := .ref address) (items := encode values)
        (by simpa only [HeapCalls.sequenceValue] using tuple_sequence heap address values found)
        (by simp [encode, present])
      change HeapMemoCalls.BlockReturnsRel module58 bases_loop.spec bound
        (.ifThen primeExpression (HeapCalls.sumBody (.constant (.int 1)) []) [] ::
          [Calls.loopContinuation filteredBody (.sequenceStepLoop "x" (.ref address) (position + 1) filteredBody []),
           .return (.name HeapCalls.sumAccumulator)]) heap
        (ScalarPost heap ((initial + modelCount (n :: remaining) : Nat) : Int)) globals
      apply HeapMemoCalls.BlockReturnsRel.branch tested
      rintro value middle ⟨valueEq, extra, shape⟩
      subst value
      have middleFound : middle[address]? = some (.tuple (encode values)) := by
        rw [shape]
        exact HeapCollection.prefix_found heap extra address _ found
      cases accepted : MillerRabin.isPrimeModel n with
      | true =>
          refine ⟨true, rfl, ?_⟩
          have countStep : modelCount (n :: remaining) = modelCount remaining + 1 := by
            simp [modelCount, accepted]
          rw [countStep]
          have increment : HeapMemoCalls.Evaluates module58 bases_loop.spec bound
              (.binOp (.name HeapCalls.sumAccumulator) .add (.constant (.int 1))) middle
              (natural (initial + 1)) middle globals :=
            HeapMemoCalls.Evaluates.binary (HeapMemoCalls.Evaluates.lookup boundAccumulator)
              HeapMemoCalls.Evaluates.constant rfl
          apply HeapMemoCalls.BlockReturnsRel.assign increment.rel
          rintro right afterConstant ⟨rightEq, afterEq⟩
          subst right
          subst afterConstant
          let updated := HeapCalls.setLocal bound HeapCalls.sumAccumulator (natural (initial + 1))
          have updatedAccumulator : updated HeapCalls.sumAccumulator = some (natural (initial + 1)) := by
            simp [updated, HeapCalls.setLocal]
          have rest := ih (position + 1) nextSuffix middle middleFound updated (initial + 1) updatedAccumulator
          have framed := rest.consequence (fun value after holds => ScalarPost.frame shape holds)
          simpa [filteredBody, HeapCalls.sumBody, Calls.loopContinuation, Calls.loopControl,
            Calls.loopControlStmt, modelCount, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using framed
      | false =>
          refine ⟨false, rfl, ?_⟩
          have countStep : modelCount (n :: remaining) = modelCount remaining := by
            simp [modelCount, accepted]
          rw [countStep]
          have rest := ih (position + 1) nextSuffix middle middleFound bound initial boundAccumulator
          have framed := rest.consequence (fun value after holds => ScalarPost.frame shape holds)
          simpa [filteredBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using framed

/-- The original generator expression counts its supplied tuple lazily, including every source-helper heap effect. -/
theorem generator_result (values : List Nat) (address : Nat) (heap : Mutable.Heap)
    (found : heap[address]? = some (.tuple (encode values))) (env : HeapCalls.Env)
    (getCorners : env "corners" = some (.ref address)) (globals : Mutable.Globals := []) :
    HeapMemoCalls.EvaluatesRel module58 bases_loop.spec env sumExpression heap
      (ScalarPost heap (modelCount values)) globals := by
  apply HeapMemoCalls.EvaluatesRel.sum_generator_sequence
    (HeapMemoCalls.EvaluatesRel.lookup getCorners
      (post := fun value after => value = .ref address ∧ after = heap) ⟨rfl, rfl⟩)
    (by intros; simp)
  · rintro value middle ⟨valueEq, heapEq⟩
    subst value
    subst middle
    exact ⟨encode values, tuple_sequence heap address values found⟩
  · intro value middle holds
    exact HeapMemoCalls.EvaluatesRel.constant
      (post := fun value after => value = number 0 ∧ after = middle) ⟨rfl, rfl⟩
  · rintro value middle initial afterStart ⟨valueEq, heapEq⟩ ⟨initialEq, afterEq⟩
    subst value
    subst middle
    subst initial
    subst afterStart
    refine ⟨rfl, ?_⟩
    simpa [filteredBody, HeapCalls.sequenceSource] using generator_loop values values 0 (by simp) address heap found
      (HeapCalls.setLocal
        (HeapCalls.comprehensionEnv env [("x", .name "corners", [primeExpression])])
        HeapCalls.sumAccumulator (natural 0)) 0 (by simp [HeapCalls.setLocal]) globals

end solvers.«58».solve

namespace solvers.«58».solve
open FormalSW.PyAst
open ProjectEulerSolutions
private abbrev layerModule := Solvers.P58Ast.module
private abbrev layerNumber (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev layerNatural (value : Nat) : Mutable.Value := .scalar (.int value)

def signedSide (layer : Nat) : Int := 2 * (layer : Int) + 1
def signedStep (layer : Nat) : Int := signedSide layer - 1
def signedSquare (layer : Nat) : Int := signedSide layer * signedSide layer

theorem square_cast (layer : Nat) : (SpiralPrimes.square layer : Int) = signedSquare layer := by
  simp [SpiralPrimes.square, SpiralPrimes.side, signedSquare, signedSide, pow_two]
theorem step_cast (layer : Nat) : (SpiralPrimes.step layer : Int) = signedStep layer := by
  simp [SpiralPrimes.step, signedStep, signedSide]
theorem corner_cast (layer offset : Nat) (bound : offset ≤ 3) :
    (SpiralPrimes.corner layer offset : Int) = signedSquare layer - offset * signedStep layer := by
  rw [SpiralPrimes.corner, Nat.cast_sub (Nat.le_of_lt (SpiralPrimes.corner_offset_lt_square layer offset bound)),
    Nat.cast_mul, square_cast, step_cast]

theorem encoded_corners (layer : Nat) : encode (SpiralPrimes.corners layer) =
    [layerNumber (signedSquare layer - signedStep layer),
     layerNumber (signedSquare layer - 2 * signedStep layer),
     layerNumber (signedSquare layer - 3 * signedStep layer)] := by
  simp [encode, SpiralPrimes.corners, corner_cast layer 1 (by decide), corner_cast layer 2 (by decide),
    corner_cast layer 3 (by decide)]

def cornerExpression : Expr := .tuple
  [.binOp (.name "sq") .sub (.name "step"),
   .binOp (.name "sq") .sub (.binOp (.constant (.int 2)) .mult (.name "step")),
   .binOp (.name "sq") .sub (.binOp (.constant (.int 3)) .mult (.name "step"))]
def layerLeading : List Stmt :=
  [.augAssign (.name "k") .add (.constant (.int 1)),
   .assign [.name "side"] (.binOp (.binOp (.constant (.int 2)) .mult (.name "k")) .add (.constant (.int 1))),
   .assign [.name "step"] (.binOp (.name "side") .sub (.constant (.int 1))),
   .assign [.name "sq"] (.binOp (.name "side") .mult (.name "side")),
   .assign [.name "corners"] cornerExpression]

structure State (env : HeapCalls.Env) (previous : Nat) : Prop where
  layer : env "k" = some (layerNatural previous)
  primes : env "prime_count" = some (layerNatural (SpiralPrimes.count previous))
  total : env "total_diag" = some (layerNatural (SpiralPrimes.total previous))
structure Ready (env : HeapCalls.Env) (previous address : Nat) : Prop where
  layer : env "k" = some (layerNatural (previous + 1))
  side : env "side" = some (layerNumber (signedSide (previous + 1)))
  step : env "step" = some (layerNumber (signedStep (previous + 1)))
  square : env "sq" = some (layerNumber (signedSquare (previous + 1)))
  corners : env "corners" = some (.ref address)
  primes : env "prime_count" = some (layerNatural (SpiralPrimes.count previous))
  total : env "total_diag" = some (layerNatural (SpiralPrimes.total previous))

private theorem corner_tuple (layer : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getSquare : env "sq" = some (layerNumber (signedSquare layer)))
    (getStep : env "step" = some (layerNumber (signedStep layer))) (globals : Mutable.Globals) :
    HeapCalls.Evaluates layerModule env cornerExpression heap (.ref heap.length)
      (heap ++ [.tuple (encode (SpiralPrimes.corners layer))]) globals := by
  simpa only [encoded_corners] using
    (HeapCalls.Evaluates.tuple (module := layerModule) (env := env) (heap := heap) (globals := globals) (.cons
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getSquare) (HeapCalls.Evaluates.lookup getStep) rfl)
      (.cons (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getSquare)
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.constant (value := .int 2))
          (HeapCalls.Evaluates.lookup getStep) rfl) rfl)
        (.cons (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getSquare)
          (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.constant (value := .int 3))
            (HeapCalls.Evaluates.lookup getStep) rfl) rfl) .nil))) (by simp))

/-- The actual first five layer statements increment k, compute its corners and allocate their tuple once. -/
theorem layer_prefix (previous : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (known : State env previous) (globals : Mutable.Globals := []) :
    ∃ readyEnv,
      HeapCalls.BlockContinues layerModule env layerLeading heap readyEnv
        (heap ++ [.tuple (encode (SpiralPrimes.corners (previous + 1)))]) globals ∧
      Ready readyEnv previous heap.length := by
  let afterK := HeapCalls.setLocal env "k" (layerNatural (previous + 1))
  let afterSide := HeapCalls.setLocal afterK "side" (layerNumber (signedSide (previous + 1)))
  let afterStep := HeapCalls.setLocal afterSide "step" (layerNumber (signedStep (previous + 1)))
  let afterSquare := HeapCalls.setLocal afterStep "sq" (layerNumber (signedSquare (previous + 1)))
  let readyEnv := HeapCalls.setLocal afterSquare "corners" (.ref heap.length)
  have increment : HeapCalls.BlockContinues layerModule env
      [.augAssign (.name "k") .add (.constant (.int 1))] heap afterK heap globals := by
    apply HeapCalls.BlockContinues.augAssign known.layer HeapCalls.Evaluates.constant
    rfl
  have sideAssigned : HeapCalls.BlockContinues layerModule afterK
      [.assign [.name "side"] (.binOp (.binOp (.constant (.int 2)) .mult (.name "k")) .add (.constant (.int 1)))]
      heap afterSide heap globals := by
    apply HeapCalls.BlockContinues.assign
    exact HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.binary HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup rfl) rfl)
      HeapCalls.Evaluates.constant rfl
  have stepAssigned : HeapCalls.BlockContinues layerModule afterSide
      [.assign [.name "step"] (.binOp (.name "side") .sub (.constant (.int 1)))] heap afterStep heap globals := by
    apply HeapCalls.BlockContinues.assign
    exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant rfl
  have squareAssigned : HeapCalls.BlockContinues layerModule afterStep
      [.assign [.name "sq"] (.binOp (.name "side") .mult (.name "side"))] heap afterSquare heap globals := by
    apply HeapCalls.BlockContinues.assign
    exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) (HeapCalls.Evaluates.lookup rfl) rfl
  have cornersAssigned : HeapCalls.BlockContinues layerModule afterSquare
      [.assign [.name "corners"] cornerExpression] heap readyEnv
      (heap ++ [.tuple (encode (SpiralPrimes.corners (previous + 1)))]) globals :=
    HeapCalls.BlockContinues.assign (corner_tuple (previous + 1) afterSquare heap rfl rfl globals)
  refine ⟨readyEnv, ?_, rfl, rfl, rfl, rfl, rfl, known.primes, known.total⟩
  simpa only [layerLeading, List.append_assoc, List.nil_append] using
    increment.append (sideAssigned.append (stepAssigned.append (squareAssigned.append cornersAssigned)))

end solvers.«58».solve

namespace solvers.«58».solve
open FormalSW.PyAst
open ProjectEulerSolutions
private abbrev finishModule := Solvers.P58Ast.module
private abbrev finishNatural (value : Nat) : Mutable.Value := .scalar (.int value)
private abbrev finishFlag (value : Bool) : Mutable.Value := .scalar (.bool value)

def ratioExpression : Expr :=
  .compare (.binOp (.name "prime_count") .mult (.constant (.int 10))) [(.lt, .name "total_diag")]

def layerTrailing : List Stmt :=
  [.augAssign (.name "prime_count") .add sumExpression,
   .augAssign (.name "total_diag") .add (.constant (.int 4)),
   .ifThen ratioExpression [.return (.name "side")] []]
def loopBody : List Stmt := layerLeading ++ layerTrailing
def whileStatement : Stmt := .whileDo (.constant (.bool true)) loopBody []

theorem source_while :
    (executableBody ((findFunctionIn finishModule.body "solve").getD ([], [])).2)[3]? = some whileStatement := rfl

private theorem finish_disjoint : HeapMemoCalls.CacheDisjoint finishModule finishModule := by
  intro name cached
  change name ∈ ([] : List String) at cached
  simp at cached

/-- The source's strict integer ratio comparison is exactly the common finite crossing predicate. -/
theorem ratio_evaluates (layer : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (known : State env layer) (globals : Mutable.Globals := []) :
    HeapMemoCalls.Evaluates finishModule bases_loop.spec env ratioExpression heap
      (finishFlag (decide (SpiralPrimes.below 1 10 layer))) heap globals := by
  have multiplied : HeapCalls.Evaluates finishModule env
      (.binOp (.name "prime_count") .mult (.constant (.int 10))) heap
      (finishNatural (SpiralPrimes.count layer * 10)) heap globals :=
    HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup known.primes) HeapCalls.Evaluates.constant rfl
  have plain : HeapCalls.Evaluates finishModule env ratioExpression heap
      (finishFlag (decide (SpiralPrimes.below 1 10 layer))) heap globals := by
    apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap) multiplied
      (HeapCalls.Evaluates.lookup known.total)
    simp [HeapCalls.compare, finishFlag, scalarCompare, evalCmpOp, SpiralPrimes.below,
      Nat.mul_comm, -Int.natCast_mul]
    omega
  exact HeapMemoCalls.Evaluates.from_heap (HeapCalls.Extends.refl _) finish_disjoint plain
    HeapMemoCalls.Cache.Preserves.refl

/-- The actual augmented count update saves its old value before the lazy generator mutates private heap state. -/
theorem update_counts (previous address : Nat) (agrees : SpiralPrimes.modelRow (previous + 1) = SpiralPrimes.row (previous + 1))
    (env : HeapCalls.Env) (heap : Mutable.Heap) (known : Ready env previous address)
    (found : heap[address]? = some (.tuple (encode (SpiralPrimes.corners (previous + 1)))))
    (following : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (continued : ∀ afterEnv afterHeap,
      State afterEnv (previous + 1) →
      afterEnv "side" = some (.scalar (.int (signedSide (previous + 1)))) →
      (∃ extra, afterHeap = heap ++ extra) →
      HeapMemoCalls.BlockReturnsRel finishModule bases_loop.spec afterEnv following afterHeap post globals) :
    HeapMemoCalls.BlockReturnsRel finishModule bases_loop.spec env
      (.augAssign (.name "prime_count") .add sumExpression ::
        .augAssign (.name "total_diag") .add (.constant (.int 4)) :: following) heap post globals := by
  have counted := generator_result (SpiralPrimes.corners (previous + 1)) address heap found env known.corners globals
  apply HeapMemoCalls.BlockReturnsRel.augAssign known.primes counted
  rintro right middle ⟨rightEq, extra, shape⟩
  subst right
  let nextCount := SpiralPrimes.count previous + modelCount (SpiralPrimes.corners (previous + 1))
  have nextCount_eq : nextCount = SpiralPrimes.count (previous + 1) := by
    change SpiralPrimes.count previous + SpiralPrimes.modelRow (previous + 1) = _
    rw [agrees, SpiralPrimes.count_succ]
  let primed := HeapCalls.setLocal env "prime_count" (finishNatural nextCount)
  refine ⟨finishNatural nextCount, middle, rfl, HeapMemoCalls.Cache.Preserves.refl, ?_⟩
  have primedTotal : primed "total_diag" = some (finishNatural (SpiralPrimes.total previous)) := known.total
  apply HeapMemoCalls.BlockReturnsRel.augAssign primedTotal
    (HeapMemoCalls.EvaluatesRel.constant
      (post := fun value after => value = finishNatural 4 ∧ after = middle) ⟨rfl, rfl⟩)
  rintro right afterConstant ⟨rightEq, heapEq⟩
  subst right
  subst afterConstant
  let nextTotal := SpiralPrimes.total previous + 4
  have nextTotal_eq : nextTotal = SpiralPrimes.total (previous + 1) := by
    dsimp [nextTotal, SpiralPrimes.total]
    omega
  let updated := HeapCalls.setLocal primed "total_diag" (finishNatural nextTotal)
  refine ⟨finishNatural nextTotal, middle, rfl, HeapMemoCalls.Cache.Preserves.refl, ?_⟩
  apply continued updated middle
  · refine ⟨known.layer, ?_, ?_⟩
    · change some (finishNatural nextCount) = some (finishNatural (SpiralPrimes.count (previous + 1)))
      rw [nextCount_eq]
    · change some (finishNatural nextTotal) = some (finishNatural (SpiralPrimes.total (previous + 1)))
      rw [nextTotal_eq]
  · exact known.side
  · exact ⟨extra, shape⟩

end solvers.«58».solve

namespace solvers.«58».solve
open FormalSW.PyAst
open ProjectEulerSolutions
private abbrev solveModule := Solvers.P58Ast.module
private abbrev solveNumber (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev solveFlag (value : Bool) : Mutable.Value := .scalar (.bool value)

private theorem solve_disjoint : HeapMemoCalls.CacheDisjoint solveModule solveModule := by
  intro name cached
  change name ∈ ([] : List String) at cached
  simp at cached
private theorem side_cast (layer : Nat) : (SpiralPrimes.side layer : Int) = signedSide layer := by
  simp [SpiralPrimes.side, signedSide]

/-- Finite remaining layers justify termination of the actual unbounded source while loop. -/
theorem loop (bound : Nat) (agreement : SpiralPrimes.RowAgreement 0 bound)
    (first : SpiralPrimes.FirstCrossing 1 10 bound)
    (remaining previous : Nat) (equation : previous + remaining = bound) (before : previous < bound)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (known : State env previous)
    (globals : Mutable.Globals := []) :
    HeapMemoCalls.BlockReturnsRel solveModule bases_loop.spec env [whileStatement] heap
      (ScalarPost heap (SpiralPrimes.side bound)) globals := by
  induction remaining generalizing previous env heap with
  | zero => omega
  | succ remaining ih =>
      apply HeapMemoCalls.BlockReturnsRel.while_loop
        (HeapMemoCalls.EvaluatesRel.constant
          (post := fun value after => value = solveFlag true ∧ after = heap) ⟨rfl, rfl⟩)
      rintro value testedHeap ⟨valueEq, heapEq⟩
      subst value
      subst testedHeap
      refine ⟨true, rfl, ?_⟩
      obtain ⟨readyEnv, leadingRun, ready⟩ := layer_prefix previous env heap known globals
      let layerHeap := heap ++ [.tuple (encode (SpiralPrimes.corners (previous + 1)))]
      have cornersFound : layerHeap[heap.length]? = some (.tuple (encode (SpiralPrimes.corners (previous + 1)))) := by
        simp [layerHeap]
      change HeapMemoCalls.BlockReturnsRel solveModule bases_loop.spec env
        (layerLeading ++ [.augAssign (.name "prime_count") .add sumExpression,
          .augAssign (.name "total_diag") .add (.constant (.int 4)),
          .ifThen ratioExpression [.return (.name "side")] [],
          Calls.loopContinuation loopBody whileStatement]) heap
        (ScalarPost heap (SpiralPrimes.side bound)) globals
      apply HeapMemoCalls.BlockReturnsRel.after_heap_prefix_frame solve_disjoint leadingRun
        (bases_loop.spec_preserves_append heap [.tuple (encode (SpiralPrimes.corners (previous + 1)))])
      apply update_counts previous heap.length
        (agreement (previous + 1) (by omega) (by omega)) readyEnv layerHeap ready cornersFound
      intro afterEnv afterHeap nextState getSide extended
      obtain ⟨extra, extended⟩ := extended
      have fromOriginal : afterHeap = heap ++ ([.tuple (encode (SpiralPrimes.corners (previous + 1)))] ++ extra) := by
        simpa only [layerHeap, List.append_assoc] using extended
      apply HeapMemoCalls.BlockReturnsRel.branch (ratio_evaluates (previous + 1) afterEnv afterHeap nextState globals).rel
      rintro value afterTest ⟨valueEq, heapEq⟩
      subst value
      subst afterTest
      by_cases crossing : SpiralPrimes.below 1 10 (previous + 1)
      · have last : previous + 1 = bound := by
          by_contra different
          exact (first.earlier (previous + 1) (by omega) (by omega)) crossing
        refine ⟨true, by simp [Mutable.truth, scalarTruth, crossing], ?_⟩
        apply HeapMemoCalls.BlockReturnsRel.return_value
        apply HeapMemoCalls.EvaluatesRel.lookup getSide
        refine ⟨?_, [.tuple (encode (SpiralPrimes.corners (previous + 1)))] ++ extra, fromOriginal⟩
        rw [last]
        exact congrArg (fun value : Int => Mutable.Value.scalar (.int value)) (side_cast bound).symm
      · have beforeNext : previous + 1 < bound := by
          by_contra noEarlier
          have last : previous + 1 = bound := by omega
          exact crossing (by simpa only [last] using first.crossing)
        refine ⟨false, by simp [Mutable.truth, scalarTruth, crossing], ?_⟩
        have rest := ih (previous + 1) (by omega) beforeNext afterEnv afterHeap nextState
        have framed := rest.consequence (fun value after holds => ScalarPost.frame fromOriginal holds)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          loopBody, layerLeading, layerTrailing] using framed

/-- The unchanged root solve() executes through the certified first crossing and returns its side length. -/
theorem result (bound : Nat) (agreement : SpiralPrimes.RowAgreement 0 bound)
    (first : SpiralPrimes.FirstCrossing 1 10 bound) (heap : Mutable.Heap)
    (globals : Mutable.Globals := []) :
    HeapMemoCalls.ReturnsRel solveModule bases_loop.spec "solve" [] heap
      (ScalarPost heap (SpiralPrimes.side bound)) globals := by
  let params := ((findFunctionIn solveModule.body "solve").getD ([], [])).1
  let sourceBody := ((findFunctionIn solveModule.body "solve").getD ([], [])).2
  let env := HeapCalls.functionEnv params sourceBody [] globals
  let primed := HeapCalls.setLocal env "prime_count" (solveNumber 0)
  let totaled := HeapCalls.setLocal primed "total_diag" (solveNumber 1)
  let initialized := HeapCalls.setLocal totaled "k" (solveNumber 0)
  apply HeapMemoCalls.BlockReturnsRel.function (by decide) rfl rfl
  apply HeapMemoCalls.BlockReturnsRel.assign (HeapMemoCalls.Evaluates.constant (value := .int 0)).rel
  rintro value afterPrime ⟨valueEq, heapEq⟩
  subst value
  subst afterPrime
  apply HeapMemoCalls.BlockReturnsRel.assign (HeapMemoCalls.Evaluates.constant (value := .int 1)).rel
  rintro value afterTotal ⟨valueEq, heapEq⟩
  subst value
  subst afterTotal
  apply HeapMemoCalls.BlockReturnsRel.assign (HeapMemoCalls.Evaluates.constant (value := .int 0)).rel
  rintro value afterLayer ⟨valueEq, heapEq⟩
  subst value
  subst afterLayer
  change HeapMemoCalls.BlockReturnsRel solveModule bases_loop.spec initialized [whileStatement] heap
    (ScalarPost heap (SpiralPrimes.side bound)) globals
  exact loop bound agreement first bound 0 (by omega) first.positive initialized heap ⟨rfl, rfl, rfl⟩ globals

end solvers.«58».solve

namespace solvers.«58».is_prime
open FormalSW.PyAst

/-- Every signed input terminates through the actual source routine. -/
theorem «Returns» (n : Int) :
    (HeapMemoCalls.scalarFunction Solvers.P58Ast.module "is_prime" [.int n]).Returns := by
  by_cases nonnegative : 0 ≤ n
  · have returned :
        (HeapMemoCalls.scalarFunction Solvers.P58Ast.module "is_prime" [.int (n.toNat : Int)]).Returns :=
      HeapMemoCalls.ReturnsRel.scalar_returns ((result n.toNat []).consequence
        (by intro value after holds; exact ⟨holds.1, True.intro⟩))
    simpa only [Int.toNat_of_nonneg nonnegative] using returned
  · exact HeapMemoCalls.ReturnsRel.scalar_returns ((small_result n (by omega) []).rel.consequence
      (by intro value after holds; exact ⟨holds.1, True.intro⟩))

theorem «Supported» (n : Int) :
    (HeapMemoCalls.scalarFunction Solvers.P58Ast.module "is_prime" [.int n]).Supported :=
  («Returns» n).supported

end solvers.«58».is_prime

namespace solvers.«58».solve
open FormalSW.PyAst

/-- Every original source operation reaches the checked side length. -/
theorem final_result (heap : Mutable.Heap) :
    HeapMemoCalls.ReturnsRel Solvers.P58Ast.module bases_loop.spec "solve" [] heap
      (fun value after => value = .scalar (.int 26241) ∧ ∃ extra, after = heap ++ extra) := by
  simpa only [ScalarPost, ProjectEulerSolutions.SpiralPrimes.side] using
    result 13120 ProjectEulerSolutions.P58Numerical.model_rows_agree
      ProjectEulerSolutions.P58Numerical.first_crossing heap

theorem «Returns» : (HeapMemoCalls.scalarFunction Solvers.P58Ast.module "solve" []).Returns :=
  HeapMemoCalls.ReturnsRel.scalar_returns ((final_result []).consequence
    (by intro value after holds; exact ⟨holds.1, True.intro⟩))

/-- Both original Python programs, including their private caches and allocations, return the same scalar. -/
theorem «Equiv(naive_default)» :
    (HeapMemoCalls.scalarFunction Solvers.P58Ast.module "solve" []).Equiv
      (HeapMemoCalls.scalarFunction P58Ast.module "naive_default" []) := by
  obtain ⟨_, _, optimized⟩ := HeapMemoCalls.ReturnsRel.scalar_finishes
    (values := []) (result := .int 26241) (post := fun _ => True)
    ((final_result []).consequence (by intro value after holds; exact ⟨holds.1, True.intro⟩))
  obtain ⟨_, _, reference⟩ := HeapMemoCalls.ReturnsRel.scalar_finishes
    (values := []) (result := .int 26241) (post := fun _ => True)
    ((data.«project-euler-statements».data.py.project_euler_statements.p58.naive_default.final_result []).consequence
      (by intro value after holds; exact ⟨holds.1, True.intro⟩))
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (HeapMemoCalls.scalarFunction Solvers.P58Ast.module "solve" []).Supported ∧
      (HeapMemoCalls.scalarFunction P58Ast.module "naive_default" []).Supported :=
  ⟨«Returns».supported,
    data.«project-euler-statements».data.py.project_euler_statements.p58.naive_default.«Supported»⟩

end solvers.«58».solve
