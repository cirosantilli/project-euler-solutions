import ProjectEulerSolutions.Lib.MillerRabin
import «.formalsw».solvers.«60.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.DecimalShift

/-! Actual pow10_for source, for every signed Int and arbitrary caller state. -/
namespace solvers.«60».pow10_for
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.DecimalShift
set_option Elab.async false

private abbrev program := Solvers.P60Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def condition : Expr := .compare (.name "p") [(.ltE, .name "n")]
def body : List Stmt := [.augAssign (.name "p") .mult (.constant (.int 10))]
def code : List Stmt :=
  [.assign [.name "p"] (.constant (.int 1)), .whileDo condition body [], .return (.name "p")]

/-- The source anchor reads the complete executable function after marker/docstring removal. -/
theorem code_source :
    executableBody ((findFunctionIn program.body "pow10_for").getD ([], [])).2 = code := rfl

/-- The actual while loop terminates by its remaining positive gap, independently
of the decimal-digit specification. Multiplication changes only the local p. -/
theorem loop (n : Int) (p : Nat) (positive : 0 < p) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) (getP : env "p" = some (number p))
    (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    BlockReturns program env [.whileDo condition body [], .return (.name "p")]
      heap (number (decimalShiftFrom n p positive)) heap globals := by
  by_cases active : (p : Int) ≤ n
  · have nextPositive : 0 < p * 10 := by omega
    let nextEnv := HeapCalls.setLocal env "p" (number (p * 10))
    have nextN : nextEnv "n" = some (number n) := by
      simpa [nextEnv, HeapCalls.setLocal] using getN
    have nextP : nextEnv "p" = some (number (p * 10)) := by simp [nextEnv, HeapCalls.setLocal]
    have continued := loop n (p * 10) nextPositive nextEnv nextN nextP heap globals
    rw [from_step n p positive active]
    apply BlockReturns.while_loop (flag := true) (value := flag true) (middle := heap)
      (Evaluates.compare (Evaluates.lookup getP) (Evaluates.lookup getN)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, active])) rfl
    apply BlockReturns.augAssign getP Evaluates.constant
      (value := number (p * 10)) (afterOperation := heap)
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
    · exact continued
  · rw [from_done n p positive (by omega)]
    apply BlockReturns.while_loop (flag := false) (value := flag false) (middle := heap)
      (Evaluates.compare (Evaluates.lookup getP) (Evaluates.lookup getN)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, active])) rfl
    exact BlockReturns.return_value (Evaluates.lookup getP)
termination_by n.toNat + 1 - p
decreasing_by exact measure_step n p positive active

/-- Full actual Python function result, without a nonnegative-input precondition. -/
theorem result (n : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "pow10_for" [number n] heap (number (decimalShift n)) heap globals := by
  let params := ((findFunctionIn program.body "pow10_for").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "pow10_for").getD ([], [])).2
  let env := HeapCalls.functionEnv params sourceBody [number n] globals
  have source : executableBody sourceBody = code := code_source
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (executableBody sourceBody) heap (number (decimalShift n)) heap globals
  rw [source]
  apply BlockReturns.assign Evaluates.constant
  exact loop n 1 (by decide) (HeapCalls.setLocal env "p" (number 1))
    (by cases globals <;> rfl) (by rfl) heap globals

theorem heapReturns (n : Int) : (HeapCalls.scalarFunction program "pow10_for" [.int n]).Returns :=
  ReturnsValue.scalar_returns (result n [])

theorem heapSupported (n : Int) : (HeapCalls.scalarFunction program "pow10_for" [.int n]).Supported :=
  (heapReturns n).supported

/-- The exact source value is the least positive decimal power strictly above n. -/
theorem least_decimal_power (n : Int) :
    ∃ width : Nat, decimalShift n = 10 ^ width ∧ n < (decimalShift n : Int) ∧
      ∀ other : Nat, n < ((10 ^ other : Nat) : Int) → decimalShift n ≤ 10 ^ other := by
  obtain ⟨width, attained, exceeds, _⟩ := decimalShift_spec n
  exact ⟨width, attained, exceeds, decimalShift_le_power n⟩

end solvers.«60».pow10_for

namespace solvers.«60».pow10_for.pure
open FormalSW.PyAst
open ProjectEulerSolutions.DecimalShift
set_option Elab.async false
private abbrev program := Solvers.P60Ast.module

/-- Scalar-call execution follows the same actual source and well-founded gap.
This supplies the runtime selected for pow10_for's public standalone marker. -/
theorem loop (n : Int) (p : Nat) (positive : 0 < p) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) (getP : env "p" = some (.int p)) :
    Calls.BlockReturns program env [.whileDo condition body [], .return (.name "p")]
      (.int (decimalShiftFrom n p positive)) := by
  by_cases active : (p : Int) ≤ n
  · have nextPositive : 0 < p * 10 := by omega
    let nextEnv := env.set "p" (.int (p * 10))
    have nextN : nextEnv "n" = some (.int n) := by
      simpa [nextEnv, ScalarEnv.set] using getN
    have nextP : nextEnv "p" = some (.int (p * 10)) := by simp [nextEnv, ScalarEnv.set]
    have continued := loop n (p * 10) nextPositive nextEnv nextN nextP
    rw [from_step n p positive active]
    apply Calls.BlockReturns.while_loop (value := .bool true)
      (Calls.Evaluates.compare (Calls.Evaluates.lookup getP) (Calls.Evaluates.lookup getN)
        (by simp [scalarCompare, evalCmpOp, active]))
    apply Calls.BlockReturns.augAssign (value := .int ((p * 10 : Nat) : Int))
      (Calls.Evaluates.binary (Calls.Evaluates.lookup getP) Calls.Evaluates.constant
        (by simp [scalarBinary]))
    exact continued
  · rw [from_done n p positive (by omega)]
    apply Calls.BlockReturns.while_loop (value := .bool false)
      (Calls.Evaluates.compare (Calls.Evaluates.lookup getP) (Calls.Evaluates.lookup getN)
        (by simp [scalarCompare, evalCmpOp, active]))
    exact Calls.BlockReturns.return_value (Calls.Evaluates.lookup getP)
termination_by n.toNat + 1 - p
decreasing_by exact measure_step n p positive active

/-- Exact pure-call result for every signed integer input. -/
theorem result (n : Int) :
    Calls.ReturnsValue program "pow10_for" [.int n] (.int (decimalShift n)) := by
  let params := ((findFunctionIn program.body "pow10_for").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "pow10_for").getD ([], [])).2
  let env := FormalSW.PyAst.argumentEnv params [.int n]
  have source : executableBody sourceBody = code := code_source
  apply Calls.BlockReturns.function rfl rfl
  change Calls.BlockReturns program env (executableBody sourceBody) (.int (decimalShift n))
  rw [source]
  apply Calls.BlockReturns.assign Calls.Evaluates.constant
  exact loop n 1 (by decide) (env.set "p" (.int 1)) (by rfl) (by rfl)

end solvers.«60».pow10_for.pure

namespace solvers.«60».pow10_for
open FormalSW.PyAst

theorem callReturns (n : Int) : (Calls.function Solvers.P60Ast.module "pow10_for" [.int n]).Returns :=
  (pure.result n).returns

theorem callSupported (n : Int) : (Calls.function Solvers.P60Ast.module "pow10_for" [.int n]).Supported :=
  (callReturns n).supported

end solvers.«60».pow10_for

-- Append after the existing heap and Calls proofs. Rename only the previous
-- Calls public adapters to callReturns/callSupported before adding these new
-- public scalarFunction adapters. The existing heap result stays unchanged.
namespace solvers.«60».pow10_for.legacy
open FormalSW.PyAst
open ProjectEulerSolutions.DecimalShift
set_option Elab.async false
private abbrev program := Solvers.P60Ast.module

/-- The original small statement interpreter performs one guard step and one
multiplication step per active iteration; the same natural gap strictly falls. -/
theorem loop (n : Int) (p : Nat) (positive : 0 < p) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) (getP : env "p" = some (.int p)) :
    Finishes scalarStep
      (⟨[.whileDo condition body [], .return (.name "p")], env⟩ : ScalarState)
      (.ok (.int (decimalShiftFrom n p positive))) := by
  by_cases active : (p : Int) ≤ n
  · have nextPositive : 0 < p * 10 := by omega
    let nextEnv := env.set "p" (.int ((p * 10 : Nat) : Int))
    have nextN : nextEnv "n" = some (.int n) := by
      simpa [nextEnv, ScalarEnv.set] using getN
    have nextP : nextEnv "p" = some (.int ((p * 10 : Nat) : Int)) := by simp [nextEnv, ScalarEnv.set]
    have continued := loop n (p * 10) nextPositive nextEnv nextN nextP
    have tested : scalarExpr env condition = some (.ok (.bool true)) := by
      simp [condition, scalarExpr, getP, getN, scalarCompare, evalCmpOp, active]
    have multiplied : scalarExpr env (.binOp (.name "p") .mult (.constant (.int 10))) =
        some (.ok (.int ((p * 10 : Nat) : Int))) := by
      simp [scalarExpr, getP, scalarBinary]
    rw [from_step n p positive active]
    apply Finishes.next (following :=
      ⟨[.augAssign (.name "p") .mult (.constant (.int 10)),
        .whileDo condition body [], .return (.name "p")], env⟩)
    · simp [scalarStep, tested, scalarTruth, body]
    · apply Finishes.next (following := ⟨[.whileDo condition body [], .return (.name "p")], nextEnv⟩)
      · simp only [scalarStep, multiplied, nextEnv]
      · exact continued
  · have tested : scalarExpr env condition = some (.ok (.bool false)) := by
      simp [condition, scalarExpr, getP, getN, scalarCompare, evalCmpOp, active]
    rw [from_done n p positive (by omega)]
    apply Finishes.next (following := ⟨[.return (.name "p")], env⟩)
    · simp [scalarStep, tested, scalarTruth]
    · exact Finishes.done (by simp [scalarStep, scalarExpr, getP])
termination_by n.toNat + 1 - p
decreasing_by exact measure_step n p positive active

/-- Actual source termination/result in the precise legacy runtime chosen by
standalone frontend coverage. This does not assume a Calls-to-scalar bridge. -/
theorem result (n : Int) :
    Finishes (FormalSW.PyAst.scalarFunction program "pow10_for" [.int n]).step
      (FormalSW.PyAst.scalarFunction program "pow10_for" [.int n]).initial (.ok (.int (decimalShift n))) := by
  let params := ((findFunctionIn program.body "pow10_for").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "pow10_for").getD ([], [])).2
  let env := FormalSW.PyAst.argumentEnv params [.int n]
  have source : executableBody sourceBody = code := code_source
  change Finishes scalarStep (⟨executableBody sourceBody, env⟩ : ScalarState) (.ok (.int (decimalShift n)))
  rw [source]
  apply Finishes.next (following :=
    ⟨[.whileDo condition body [], .return (.name "p")], env.set "p" (.int 1)⟩)
  · rfl
  · exact loop n 1 (by decide) (env.set "p" (.int 1)) (by rfl) (by rfl)

end solvers.«60».pow10_for.legacy

namespace solvers.«60».pow10_for
open FormalSW.PyAst
open ProjectEulerSolutions.DecimalShift

theorem «Returns» (n : Int) : (FormalSW.PyAst.scalarFunction Solvers.P60Ast.module "pow10_for" [.int n]).Returns :=
  ⟨.int (decimalShift n), legacy.result n⟩

theorem «Supported» (n : Int) : (FormalSW.PyAst.scalarFunction Solvers.P60Ast.module "pow10_for" [.int n]).Supported :=
  («Returns» n).supported

end solvers.«60».pow10_for

set_option maxRecDepth 4096


/-! Primality source components with an explicit module-global context. -/
namespace ProjectEulerSolutions.MillerRabin

/-- P60 tests divisibility first and returns equality; P58 tests equality first.
The models agree because equality implies zero remainder, including p=0. -/
theorem smallScan_cons_divisibility_first (n : Int) (p : Nat) (rest : List Nat) :
    smallScan n (p :: rest) =
      if n % (p : Int) = 0 then some (decide (n = (p : Int))) else smallScan n rest := by
  by_cases same : n = (p : Int)
  · subst n
    simp [smallScan]
  · simp [smallScan, same]

end ProjectEulerSolutions.MillerRabin

namespace solvers.«60».is_prime
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin
private abbrev contextProgram := Solvers.P60Ast.module

def globalInitializer : List Stmt → String → Option Expr
  | [], _ => none
  | .assign [.name bound] value :: rest, name =>
      if bound = name then some value else globalInitializer rest name
  | _ :: rest, name => globalInitializer rest name

def smallValues : List Mutable.Value := smallPrimes.map (fun p => .scalar (.int p))
def baseValues : List Mutable.Value := bases.map (fun a => .scalar (.int a))

/-- Actual source facts; these do not by themselves claim initialization ran. -/
theorem bases_initializer_source :
    globalInitializer contextProgram.body "_MR_BASES_64" =
      some (.tuple (bases.map (fun a => .constant (.int a)))) := rfl

theorem small_initializer_source :
    globalInitializer contextProgram.body "_SMALL_PRIMES" =
      some (.tuple (smallPrimes.map (fun p => .constant (.int p)))) := rfl

/-- Explicit raw-call context, supplied by the source table-initialization proof
below. The table references and contents remain available to helper proofs. -/
structure SourceGlobals (heap : Mutable.Heap) (globals : Mutable.Globals) where
  smallAddress : Nat
  basesAddress : Nat
  smallBinding : globals.lookup "_SMALL_PRIMES" = some (.ref smallAddress)
  basesBinding : globals.lookup "_MR_BASES_64" = some (.ref basesAddress)
  smallTuple : heap[smallAddress]? = some (.tuple smallValues)
  basesTuple : heap[basesAddress]? = some (.tuple baseValues)

end solvers.«60».is_prime

namespace solvers.«60».is_prime.decomposition
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin

private abbrev program := Solvers.P60Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def condition : Expr :=
  .compare (.binOp (.name "d") .bitAnd (.constant (.int 1))) [(.eq, .constant (.int 0))]
def body : List Stmt :=
  [.augAssign (.name "s") .add (.constant (.int 1)),
   .augAssign (.name "d") .rightShift (.constant (.int 1))]
def statement : Stmt := .whileDo condition body []

theorem source_statement :
    (executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2)[4]? = some statement := rfl

private theorem condition_evaluates (d : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getD : env "d" = some (number d)) (globals : Mutable.Globals) :
    HeapCalls.Evaluates program env condition heap (flag (decide (d % 2 = 0))) heap globals := by
  have remainder : HeapCalls.Evaluates program env
      (.binOp (.name "d") .bitAnd (.constant (.int 1))) heap (number ((d % 2 : Nat) : Int)) heap globals := by
    apply HeapCalls.Evaluates.binary (middle := heap) (finalHeap := heap)
      (HeapCalls.Evaluates.lookup getD) HeapCalls.Evaluates.constant
    change (some (Except.ok (number (integerBitAnd d 1)), heap) :
      Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) =
      some (Except.ok (number ((d % 2 : Nat) : Int)), heap)
    rw [integerBitAnd_nat_one]
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
              [.augAssign (.name "d") .rightShift (.constant (.int 1))] heap updated heap globals := by
            apply HeapCalls.BlockContinues.augAssign incrementedD HeapCalls.Evaluates.constant
            simp [HeapCalls.inplace, Mutable.binary, number, scalarShiftRight, scalarNumber, integerShiftRight]
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

end solvers.«60».is_prime.decomposition

namespace solvers.«60».is_prime.small_prime_scan
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin
variable {globals : Mutable.Globals}
private abbrev program := Solvers.P60Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def numbers (values : List Nat) : List Mutable.Value := values.map (fun p : Nat => number (p : Int))
def equalityTest : Expr := .compare (.name "n") [(.eq, .name "p")]
def divisionTest : Expr :=
  .compare (.binOp (.name "n") .mod (.name "p")) [(.eq, .constant (.int 0))]
def body : List Stmt := [.ifThen divisionTest [.return equalityTest] []]

theorem source_statement :
    (executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2)[1]? =
      some (.forIn (.name "p") (.name "_SMALL_PRIMES") body []) := rfl

def answer (scan : Option Bool) (fallback : Mutable.Value) : Mutable.Value :=
  match scan with | some b => flag b | none => fallback
def finalHeap (scan : Option Bool) (initial fallback : Mutable.Heap) : Mutable.Heap :=
  match scan with | some _ => initial | none => fallback

/-- Traverse the live source tuple, preserving the divisible-first early return.
A miss composes the ACTUAL supplied source suffix, with all non-p locals intact. -/
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
        (by simp only [numbers, List.getElem?_map, exhausted, Option.map_none])
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
      have remainder : HeapCalls.Evaluates program bound
          (.binOp (.name "n") .mod (.name "p")) heap (number (n % (p : Int))) heap globals := by
        apply HeapCalls.Evaluates.binary (middle := heap) (finalHeap := heap)
          (HeapCalls.Evaluates.lookup boundN) (HeapCalls.Evaluates.lookup boundP)
        simp [Mutable.binary, number, scalarBinary, pPositive, Nat.ne_of_gt pPositive]
      have dividesTest : HeapCalls.Evaluates program bound divisionTest heap
          (flag (decide (n % (p : Int) = 0))) heap globals := by
        apply HeapCalls.Evaluates.compare (middle := heap) (afterRight := heap)
          (av := number (n % (p : Int))) (bv := number 0) remainder HeapCalls.Evaluates.constant
        simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
      apply HeapCalls.BlockReturns.sequence_step (value := number p) sequence
        (by simp only [numbers, List.getElem?_map, present, Option.map_some])
      simp only [smallScan_cons_divisibility_first]
      by_cases divides : n % (p : Int) = 0
      · simp only [if_pos divides, answer, finalHeap]
        apply HeapCalls.BlockReturns.branch (flag := true)
          (by simpa only [divides, decide_true] using dividesTest) rfl
        apply HeapCalls.BlockReturns.return_value
        apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup boundN) (HeapCalls.Evaluates.lookup boundP)
        simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
      · simp only [if_neg divides]
        apply HeapCalls.BlockReturns.branch (flag := false)
          (by simpa only [divides, decide_false] using dividesTest) rfl
        have rest := ih (position + 1) nextSuffix
          (fun q member => positive q (by simp [member])) bound boundN
          (fun tailMiss afterEnv unchanged => continued
            (by simpa only [smallScan_cons_divisibility_first, if_neg divides] using tailMiss)
            afterEnv (fun name notP =>
              (unchanged name notP).trans (by simp [bound, HeapCalls.setLocal, notP])))
        simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

/-- The actual global NAME resolves to the supplied actual tuple ref once.
Its origin/value hypotheses are discharged by the future initializer proof. -/
theorem result (n : Int) (values : List Nat) (positive : ∀ p ∈ values, 0 < p)
    (address : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (found : heap[address]? = some (.tuple (numbers values)))
    (getValues : env "_SMALL_PRIMES" = some (.ref address)) (getN : env "n" = some (number n))
    (following : List Stmt) (fallback : Mutable.Value) (fallbackHeap : Mutable.Heap)
    (continued : smallScan n values = none →
      ∀ afterEnv, (∀ name, name ≠ "p" → afterEnv name = env name) →
      HeapCalls.BlockReturns program afterEnv following heap fallback fallbackHeap globals) :
    HeapCalls.BlockReturns program env
      (.forIn (.name "p") (.name "_SMALL_PRIMES") body [] :: following) heap
      (answer (smallScan n values) fallback) (finalHeap (smallScan n values) heap fallbackHeap) globals := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (numbers values), heap) := by
    simp [Mutable.sequence, found]
  apply HeapCalls.BlockReturns.for_sequence (by intros; simp)
    (HeapCalls.Evaluates.lookup getValues) sequence
  exact loop n values values 0 (by simp) positive (.ref address) env heap sequence getN
    following fallback fallbackHeap continued

/-- Every known small-prime answer finishes here without executing later MR calls. -/
theorem known_result (n : Int) (answerValue : Bool) (known : smallScan n smallPrimes = some answerValue)
    (address : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (found : heap[address]? = some (.tuple (numbers smallPrimes)))
    (getValues : env "_SMALL_PRIMES" = some (.ref address)) (getN : env "n" = some (number n))
    (following : List Stmt) :
    HeapCalls.BlockReturns program env
      (.forIn (.name "p") (.name "_SMALL_PRIMES") body [] :: following)
      heap (flag answerValue) heap globals := by
  have scanned := result (globals := globals) n smallPrimes (by decide) address env heap found getValues getN
    following (.scalar .none) heap (fun impossible _ _ => by rw [known] at impossible; contradiction)
  simpa only [known, answer, finalHeap] using scanned

end solvers.«60».is_prime.small_prime_scan

namespace solvers.«60».is_prime
open FormalSW.PyAst
private abbrev smallProgram := Solvers.P60Ast.module
private abbrev smallNumber (n : Int) : Mutable.Value := .scalar (.int n)

/-- The signed small-input path does not read module tables or alter the heap. -/
theorem below_two_result (n : Int) (small : n < 2) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapCalls.ReturnsValue smallProgram "is_prime" [smallNumber n] heap (.scalar (.bool false)) heap globals := by
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.branch (flag := true) (value := .scalar (.bool true)) (middle := heap)
    (HeapCalls.Evaluates.compare (av := smallNumber n) (bv := smallNumber 2)
      (middle := heap) (afterRight := heap)
      (HeapCalls.Evaluates.lookup (by cases globals <;> rfl)) HeapCalls.Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])) rfl
  exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant

end solvers.«60».is_prime

namespace solvers.«60».is_prime.decomposition
open FormalSW.PyAst
open ProjectEulerSolutions.MillerRabin
private abbrev prefixProgram := Solvers.P60Ast.module
private abbrev prefixNumber (value : Int) : Mutable.Value := .scalar (.int value)

def initialCode : List Stmt :=
  [.assign [.name "d"] (.binOp (.name "n") .sub (.constant (.int 1))),
   .assign [.name "s"] (.constant (.int 0)), statement]

theorem initial_source :
    ((executableBody ((findFunctionIn prefixProgram.body "is_prime").getD ([], [])).2).drop 2).take 3 =
      initialCode := rfl

/-- On a small-scan miss, execute both real initial assignments and then the
actual bitwise loop. Every other local and the whole supplied heap are retained. -/
theorem initialized_result (n : Nat) (large : 2 ≤ n) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (prefixNumber n)) (globals : Mutable.Globals := []) :
    ∃ afterEnv,
      HeapCalls.BlockContinues prefixProgram env initialCode heap afterEnv heap globals ∧
      afterEnv "d" = some (prefixNumber (decompose (n - 1)).1) ∧
      afterEnv "s" = some (prefixNumber (decompose (n - 1)).2) ∧
      ∀ name, name ≠ "d" → name ≠ "s" → afterEnv name = env name := by
  let withD := HeapCalls.setLocal env "d" (prefixNumber (n - 1 : Nat))
  let ready := HeapCalls.setLocal withD "s" (prefixNumber 0)
  have decreased : HeapCalls.Evaluates prefixProgram env
      (.binOp (.name "n") .sub (.constant (.int 1))) heap (prefixNumber (n - 1 : Nat)) heap globals := by
    apply HeapCalls.Evaluates.binary (value := prefixNumber (n - 1 : Nat))
      (HeapCalls.Evaluates.lookup getN) HeapCalls.Evaluates.constant
    simp [Mutable.binary, scalarBinary, prefixNumber, Int.natCast_sub (by omega : 1 ≤ n)]
  have assignD : HeapCalls.BlockContinues prefixProgram env
      [.assign [.name "d"] (.binOp (.name "n") .sub (.constant (.int 1)))] heap withD heap globals :=
    HeapCalls.BlockContinues.assign decreased
  have assignS : HeapCalls.BlockContinues prefixProgram withD
      [.assign [.name "s"] (.constant (.int 0))] heap ready heap globals :=
    HeapCalls.BlockContinues.assign HeapCalls.Evaluates.constant
  have readyD : ready "d" = some (prefixNumber (n - 1 : Nat)) := by
    simp [ready, withD, HeapCalls.setLocal]
  have readyS : ready "s" = some (prefixNumber 0) := by simp [ready, HeapCalls.setLocal]
  obtain ⟨afterEnv, actual, afterD, afterS, unchanged⟩ :=
    result (n - 1) (by omega) ready heap readyD readyS globals
  refine ⟨afterEnv, ?_, afterD, afterS, ?_⟩
  · simpa only [initialCode, List.cons_append, List.nil_append] using
      (assignD.append assignS).append actual
  · intro name notD notS
    exact (unchanged name notD notS).trans (by simp [ready, withD, HeapCalls.setLocal, notD, notS])

end solvers.«60».is_prime.decomposition


namespace solvers.«60».is_prime.bases
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.MillerRabin
private abbrev program := Solvers.P60Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev lastExpression : Expr := .binOp (.name "n") .sub (.constant (.int 1))
private abbrev lastTest : Expr := .compare (.name "x") [(.eq, lastExpression)]
private abbrev squareExpression : Expr :=
  .binOp (.binOp (.name "x") .mult (.name "x")) .mod (.name "n")
def squareBody : List Stmt :=
  [.assign [.name "x"] squareExpression, .ifThen lastTest [.breakLoop] []]
def squareElse : List Stmt := [.return (.constant (.bool false))]

def sourceBaseBody : List Stmt :=
  match ((executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2)[5]? : Option Stmt) with
  | some (Stmt.forIn _ _ body _) => body
  | _ => []

theorem square_source : sourceBaseBody[3]? = some
    (.forIn (.name "_") (.call (.name "range") [.binOp (.name "s") .sub (.constant (.int 1))])
      squareBody squareElse) := rfl

private theorem last_test (n : Nat) (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getX : env "x" = some (number x))
    (globals : Mutable.Globals) :
    Evaluates program env lastTest heap (flag (decide (x = (n : Int) - 1))) heap globals := by
  apply Evaluates.compare (Evaluates.lookup getX)
    (Evaluates.binary (Evaluates.lookup getN) Evaluates.constant rfl)
  simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]

/-- Exhaustion executes the actual for-else return False. A hit executes break,
skips that else, and runs the supplied real outer continuation. -/
theorem square_loop (remaining n : Nat) (positive : 2 ≤ n) (next stop : Int)
    (equation : (stop - next).toNat = remaining) (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getX : env "x" = some (number x))
    (following : List Stmt) (tailAnswer : Bool)
    (continued : ∀ afterEnv, (∀ name, name ≠ "_" → name ≠ "x" → afterEnv name = env name) →
      BlockReturns program afterEnv following heap (flag tailAnswer) heap globals) :
    BlockReturns program env
      (.rangeStepLoop "_" next stop 1 squareBody squareElse :: following)
      heap (flag (squareSearch n remaining x && tailAnswer)) heap globals := by
  induction remaining generalizing next x env with
  | zero =>
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value Evaluates.constant
  | succ remaining ih =>
      let bound := setLocal env "_" (number next)
      let squared := x * x % (n : Int)
      let updated := setLocal bound "x" (number squared)
      have boundN : bound "n" = some (number n) := by simpa [bound, setLocal] using getN
      have boundX : bound "x" = some (number x) := by simpa [bound, setLocal] using getX
      have updatedN : updated "n" = some (number n) := by simpa [updated, setLocal] using boundN
      have updatedX : updated "x" = some (number squared) := by simp [updated, setLocal]
      have stable : ∀ name, name ≠ "_" → name ≠ "x" → updated name = env name := by
        intro name notIndex notX
        simp [updated, bound, setLocal, notIndex, notX]
      have evaluated : Evaluates program bound squareExpression heap (number squared) heap globals := by
        apply Evaluates.binary
          (Evaluates.binary (Evaluates.lookup boundX) (Evaluates.lookup boundX) rfl)
          (Evaluates.lookup boundN)
        simp [squared, Mutable.binary, number, scalarBinary,
          show n ≠ 0 by omega, show 0 < n by omega]
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockReturns.assign evaluated
      by_cases hit : squared = (n : Int) - 1
      · have score : squareSearch n (remaining + 1) x = true := by
          change (if squared = (n : Int) - 1 then true else squareSearch n remaining squared) = true
          rw [if_pos hit]
        rw [score]
        simp only [Bool.true_and]
        apply BlockReturns.branch (flag := true)
          (by simpa only [show decide (squared = (n : Int) - 1) = true by simp only [hit, decide_true]]
            using last_test n squared updated heap updatedN updatedX globals) rfl
        apply BlockReturns.break_loop (target := by rfl)
        exact continued updated stable
      · have score : squareSearch n (remaining + 1) x = squareSearch n remaining squared := by
          change (if squared = (n : Int) - 1 then true else squareSearch n remaining squared) = _
          rw [if_neg hit]
        rw [score]
        apply BlockReturns.branch (flag := false)
          (by simpa only [show decide (squared = (n : Int) - 1) = false by simp only [hit, decide_false]]
            using last_test n squared updated heap updatedN updatedX globals) rfl
        have rest := ih (next + 1) (by omega) squared updated updatedN updatedX
          (fun afterEnv unchanged => continued afterEnv (fun name notIndex notX =>
            (unchanged name notIndex notX).trans (stable name notIndex notX)))
        simpa [squareBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using
          BlockReturns.loop_resume rest

end solvers.«60».is_prime.bases


namespace solvers.«60».is_prime.bases
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.MillerRabin
private abbrev outerProgram := Solvers.P60Ast.module
private abbrev outerNumber (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev outerFlag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev outerLast : Expr := .binOp (.name "n") .sub (.constant (.int 1))
private abbrev outerLastTest : Expr := .compare (.name "x") [(.eq, outerLast)]
def initialTest : Expr := .boolOp .or
  [.compare (.name "x") [(.eq, .constant (.int 1))], outerLastTest]
def remainderTest : Expr :=
  .compare (.binOp (.name "a") .mod (.name "n")) [(.eq, .constant (.int 0))]
def powerExpression : Expr := .builtinCall .pow [.name "a", .name "d", .name "n"]
def baseBody : List Stmt :=
  [.ifThen remainderTest [.continueLoop] [], .assign [.name "x"] powerExpression,
   .ifThen initialTest [.continueLoop] [],
   .forIn (.name "_") (.call (.name "range") [.binOp (.name "s") .sub (.constant (.int 1))]) squareBody squareElse]

theorem base_body_source : sourceBaseBody = baseBody := rfl

def changed : List String := ["a", "x", "_"]
def Frame (before after : HeapCalls.Env) : Prop := ∀ name, name ∉ changed → after name = before name

theorem Frame.refl (env : HeapCalls.Env) : Frame env env := fun _ _ => rfl
theorem Frame.trans {first second third : HeapCalls.Env} (left : Frame first second)
    (right : Frame second third) : Frame first third := fun name outside => (right name outside).trans (left name outside)
theorem Frame.updated (env : HeapCalls.Env) (name : String) (value : Mutable.Value)
    (written : name ∈ changed) : Frame env (setLocal env name value) := by
  intro other outside
  have different : other ≠ name := by intro same; subst other; exact outside written
  simp [setLocal, different]
theorem Frame.of_square {before after : HeapCalls.Env}
    (frame : ∀ name, name ≠ "_" → name ≠ "x" → after name = before name) : Frame before after := by
  intro name outside
  apply frame name
  · intro same; subst name; exact outside (by decide)
  · intro same; subst name; exact outside (by decide)

structure Inputs (env : HeapCalls.Env) (n d s : Nat) : Prop where
  n_value : env "n" = some (outerNumber n)
  d_value : env "d" = some (outerNumber d)
  s_value : env "s" = some (outerNumber s)

theorem Inputs.frame {before after : HeapCalls.Env} {n d s : Nat} (inputs : Inputs before n d s)
    (frame : Frame before after) : Inputs after n d s :=
  ⟨(frame "n" (by decide)).trans inputs.n_value, (frame "d" (by decide)).trans inputs.d_value,
   (frame "s" (by decide)).trans inputs.s_value⟩

def numbers (values : List Int) : List Mutable.Value := values.map outerNumber

private theorem initial_test (n : Nat) (x : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (outerNumber n)) (getX : env "x" = some (outerNumber x))
    (globals : Mutable.Globals) :
    Evaluates outerProgram env initialTest heap
      (outerFlag (decide (x = 1 ∨ x = (n : Int) - 1))) heap globals := by
  have first : Evaluates outerProgram env
      (.compare (.name "x") [(.eq, .constant (.int 1))]) heap
      (outerFlag (decide (x = 1))) heap globals := by
    apply Evaluates.compare (Evaluates.lookup getX) Evaluates.constant
    simp [HeapCalls.compare, outerFlag, scalarCompare, evalCmpOp]
  have last : Evaluates outerProgram env outerLastTest heap
      (outerFlag (decide (x = (n : Int) - 1))) heap globals := by
    apply Evaluates.compare (Evaluates.lookup getX)
      (Evaluates.binary (Evaluates.lookup getN) Evaluates.constant rfl)
    simp [HeapCalls.compare, outerFlag, scalarCompare, evalCmpOp]
  by_cases one : x = 1
  · simp only [one, true_or, decide_true]
    apply Evaluates.boolean_stop (flag := true) (by simpa [one] using first) rfl rfl
  · apply Evaluates.boolean_next (flag := false) (by simpa [one] using first) rfl rfl
    apply Evaluates.boolean_last
    · simpa [one] using last
    · cases decide (x = (n : Int) - 1) <;> rfl

/-- The real source base loop executes both continue paths and the nested for-else.
Its result is the ordered conjunction of the exact modular tests and the actual continuation. -/
theorem loop (n d s : Nat) (positive : 2 ≤ n) (values remaining : List Int) (position : Nat)
    (suffix : values.drop position = remaining) (source : SequenceSource)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (sequence : Mutable.sequence (sequenceValue source) heap = some (.ok (numbers values), heap))
    (inputs : Inputs env n d s) (following : List Stmt) (tailAnswer : Bool)
    (continued : ∀ afterEnv, Frame env afterEnv →
      BlockReturns outerProgram afterEnv following heap (outerFlag tailAnswer) heap globals) :
    BlockReturns outerProgram env (.sequenceStepLoop "a" source position baseBody [] :: following) heap
      (outerFlag (remaining.all (fun a => millerCheck a n d s) && tailAnswer)) heap globals := by
  induction remaining generalizing position env with
  | nil =>
      have exhausted : values[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      apply BlockReturns.sequence_done sequence
        (by simp only [numbers, List.getElem?_map, exhausted, Option.map_none])
      exact continued env (Frame.refl env)
  | cons a remaining ih =>
      have present : values[position]? = some a := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      have nextSuffix : values.drop (position + 1) = remaining := by simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := setLocal env "a" (outerNumber a)
      let next := Stmt.sequenceStepLoop "a" source (position + 1) baseBody []
      have boundFrame : Frame env bound := Frame.updated env "a" _ (by decide)
      have boundInputs : Inputs bound n d s := inputs.frame boundFrame
      have getA : bound "a" = some (outerNumber a) := by simp [bound, setLocal]
      have nextRun (afterEnv : HeapCalls.Env) (framed : Frame env afterEnv) :
          BlockReturns outerProgram afterEnv (next :: following) heap
            (outerFlag (remaining.all (fun a => millerCheck a n d s) && tailAnswer)) heap globals :=
        ih (position + 1) nextSuffix afterEnv (inputs.frame framed)
          (fun finalEnv finalFrame => continued finalEnv (framed.trans finalFrame))
      have reduced : Evaluates outerProgram bound
          (.binOp (.name "a") .mod (.name "n")) heap (outerNumber (a % (n : Int))) heap globals := by
        apply Evaluates.binary (middle := heap) (finalHeap := heap)
          (Evaluates.lookup getA) (Evaluates.lookup boundInputs.n_value)
        simp [Mutable.binary, outerNumber, scalarBinary, show n ≠ 0 by omega, show 0 < n by omega]
      have tested : Evaluates outerProgram bound remainderTest heap
          (outerFlag (decide (a % (n : Int) = 0))) heap globals := by
        apply Evaluates.compare (middle := heap) (afterRight := heap)
          (av := outerNumber (a % (n : Int))) (bv := outerNumber 0) reduced Evaluates.constant
        simp [HeapCalls.compare, outerFlag, scalarCompare, evalCmpOp]
      apply BlockReturns.sequence_step (value := outerNumber a) sequence
        (by simp only [numbers, List.getElem?_map, present, Option.map_some])
      simp only [List.all_cons]
      by_cases divisible : a % (n : Int) = 0
      · have head : millerCheck a n d s = true := by simp [millerCheck, divisible]
        rw [head]
        simp only [Bool.true_and]
        apply BlockReturns.branch (flag := true) (by simpa [divisible] using tested) rfl
        apply BlockReturns.continue_loop (target := by rfl)
        exact nextRun bound boundFrame
      · apply BlockReturns.branch (flag := false) (by simpa [divisible] using tested) rfl
        let x := FormalSW.Py.Builtin.Int.powModBinary a d n
        have power : Evaluates outerProgram bound powerExpression heap (outerNumber x) heap globals := by
          apply Evaluates.builtin (.cons (Evaluates.lookup getA) (.cons (Evaluates.lookup boundInputs.d_value)
            (.cons (Evaluates.lookup boundInputs.n_value) .nil)))
          simp [HeapCalls.primitive, HeapCalls.scalarResult, Calls.modularPower, scalarNumber,
            FormalSW.Py.Builtin.Int.powMod_nonnegative a d n (by omega),
            ← FormalSW.Py.Builtin.Int.powModBinary_correct a d n, outerNumber, x]
        apply BlockReturns.assign power
        let updated := setLocal bound "x" (outerNumber x)
        have updatedFrame : Frame env updated := boundFrame.trans (Frame.updated bound "x" _ (by decide))
        have updatedInputs : Inputs updated n d s := inputs.frame updatedFrame
        have getX : updated "x" = some (outerNumber x) := by simp [updated, setLocal]
        have initial := initial_test n x updated heap updatedInputs.n_value getX globals
        by_cases accepted : x = 1 ∨ x = (n : Int) - 1
        · have head : millerCheck a n d s = true := by simp [millerCheck, divisible, x, accepted]
          rw [head]
          simp only [Bool.true_and]
          apply BlockReturns.branch (flag := true) (by simpa [accepted] using initial) rfl
          apply BlockReturns.continue_loop (target := by rfl)
          exact nextRun updated updatedFrame
        · have head : millerCheck a n d s = squareSearch n (s - 1) x := by
            rw [millerCheck, if_neg divisible]
            change (if x = 1 ∨ x = (n : Int) - 1 then true else squareSearch n (s - 1) x) = _
            rw [if_neg accepted]
          rw [head, Bool.and_assoc]
          apply BlockReturns.branch (flag := false) (by simpa [accepted] using initial) rfl
          apply BlockReturns.for_range
            (.cons (Evaluates.binary (Evaluates.lookup updatedInputs.s_value) Evaluates.constant rfl) .nil) rfl
          exact square_loop (s - 1) n positive 0 ((s : Int) - 1) (by omega) x updated heap
            updatedInputs.n_value getX (.loopResume next :: following)
            (remaining.all (fun a => millerCheck a n d s) && tailAnswer)
            (fun afterEnv squareFrame => BlockReturns.loop_resume
              (nextRun afterEnv (updatedFrame.trans (Frame.of_square squareFrame))))

/-- Evaluate the actual global sequence once, then run the source conjunction. -/
theorem result (n d s : Nat) (positive : 2 ≤ n) (values : List Int) (address : Nat)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (found : heap[address]? = some (.tuple (numbers values)))
    (getValues : env "_MR_BASES_64" = some (.ref address)) (inputs : Inputs env n d s)
    (globals : Mutable.Globals := []) :
    BlockReturns outerProgram env
      [.forIn (.name "a") (.name "_MR_BASES_64") baseBody [], .return (.constant (.bool true))] heap
      (outerFlag (values.all (fun a => millerCheck a n d s))) heap globals := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (numbers values), heap) := by
    simp [Mutable.sequence, found]
  apply BlockReturns.for_sequence (by intros; simp) (Evaluates.lookup getValues) sequence
  simpa only [List.drop_zero, Bool.and_true] using
    loop n d s positive values values 0 rfl (.ref address) env heap sequence inputs
      [.return (.constant (.bool true))] true (fun _ _ => BlockReturns.return_value Evaluates.constant)

end solvers.«60».is_prime.bases

namespace solvers.«60».is_prime
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.MillerRabin
private abbrev finalProgram := Solvers.P60Ast.module
private abbrev finalNumber (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev finalFlag (value : Bool) : Mutable.Value := .scalar (.bool value)

def basesCode : List Stmt :=
  [.forIn (.name "a") (.name "_MR_BASES_64") bases.baseBody [], .return (.constant (.bool true))]
def largeCode : List Stmt :=
  .forIn (.name "p") (.name "_SMALL_PRIMES") small_prime_scan.body [] ::
    (decomposition.initialCode ++ basesCode)
def fullCode : List Stmt :=
  .ifThen (.compare (.name "n") [(.lt, .constant (.int 2))]) [.return (.constant (.bool false))] [] :: largeCode

theorem complete_source :
    executableBody ((findFunctionIn finalProgram.body "is_prime").getD ([], [])).2 = fullCode := rfl

/-- Every actual source branch terminates and computes the exact Miller-Rabin
algorithmic model under the supplied source table context. No initializer or
unrestricted mathematical primality result is assumed. -/
theorem natural_result (n : Nat) (positive : 2 ≤ n) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (context : SourceGlobals heap globals) :
    ReturnsValue finalProgram "is_prime" [finalNumber n] heap
      (finalFlag (isPrimeModel n)) heap globals := by
  let params := ((findFunctionIn finalProgram.body "is_prime").getD ([], [])).1
  let body := ((findFunctionIn finalProgram.body "is_prime").getD ([], [])).2
  let initial := functionEnv params body [finalNumber n] globals
  have getN : initial "n" = some (finalNumber n) := by cases globals <;> rfl
  have getSmall : initial "_SMALL_PRIMES" = some (.ref context.smallAddress) := by
    cases globals with
    | nil => have impossible := context.smallBinding; cases impossible
    | cons entry rest => exact context.smallBinding
  have getBases : initial "_MR_BASES_64" = some (.ref context.basesAddress) := by
    cases globals with
    | nil => have impossible := context.basesBinding; cases impossible
    | cons entry rest => exact context.basesBinding
  have smallFound : heap[context.smallAddress]? = some (.tuple (small_prime_scan.numbers smallPrimes)) :=
    context.smallTuple
  have baseFound : heap[context.basesAddress]? = some (.tuple (bases.numbers ProjectEulerSolutions.MillerRabin.bases)) :=
    context.basesTuple
  let ds := decompose (n - 1)
  let finalAnswer := ProjectEulerSolutions.MillerRabin.bases.all (fun a => millerCheck a n ds.1 ds.2)
  have scanned := small_prime_scan.result (globals := globals) n smallPrimes (by decide)
    context.smallAddress initial heap smallFound getSmall getN
    (decomposition.initialCode ++ basesCode) (finalFlag finalAnswer) heap
    (fun _ afterEnv unchanged => by
      have afterN : afterEnv "n" = some (finalNumber n) := (unchanged "n" (by decide)).trans getN
      have afterBases : afterEnv "_MR_BASES_64" = some (.ref context.basesAddress) :=
        (unchanged "_MR_BASES_64" (by decide)).trans getBases
      obtain ⟨ready, prepared, getD, getS, framed⟩ :=
        decomposition.initialized_result n positive afterEnv heap afterN globals
      have readyN : ready "n" = some (finalNumber n) := (framed "n" (by decide) (by decide)).trans afterN
      have readyBases : ready "_MR_BASES_64" = some (.ref context.basesAddress) :=
        (framed "_MR_BASES_64" (by decide) (by decide)).trans afterBases
      have finished := bases.result n ds.1 ds.2 positive ProjectEulerSolutions.MillerRabin.bases
        context.basesAddress ready heap baseFound readyBases ⟨readyN, getD, getS⟩ globals
      exact prepared.returns finished)
  have valueEq : small_prime_scan.answer (smallScan n smallPrimes) (finalFlag finalAnswer) =
      finalFlag (isPrimeModel n) := by
    cases scan : smallScan n smallPrimes <;>
      simp only [isPrimeModel, if_neg (show ¬n < 2 by omega), scan, small_prime_scan.answer, ds, finalAnswer]
  have heapEq : small_prime_scan.finalHeap (smallScan n smallPrimes) heap heap = heap := by
    cases smallScan n smallPrimes <;> rfl
  rw [valueEq, heapEq] at scanned
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) (value := finalFlag false) (middle := heap)
    (Evaluates.compare (av := finalNumber n) (bv := finalNumber 2) (middle := heap) (afterRight := heap)
      (Evaluates.lookup getN) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, show ¬(n : Int) < 2 by omega])) rfl
  exact scanned

/-- All signed inputs are covered, with negative values following the actual
n<2 return. The raw interface retains its explicit module context. -/
theorem raw_result (n : Int) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (context : SourceGlobals heap globals) :
    ReturnsValue finalProgram "is_prime" [finalNumber n] heap
      (finalFlag (isPrimeModel n.toNat)) heap globals := by
  by_cases small : n < 2
  · have modelSmall : n.toNat < 2 := by omega
    simpa only [isPrimeModel, if_pos modelSmall] using below_two_result n small heap globals
  · have nonnegative : 0 ≤ n := by omega
    simpa only [Int.toNat_of_nonneg nonnegative] using natural_result n.toNat (by omega) heap globals context

end solvers.«60».is_prime

namespace solvers.«60».is_prime.initialization
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.MillerRabin

private abbrev program := Solvers.P60Ast.module

/-- The two real module assignments, in source order. Imports and function
definitions preceding them do not form part of this data-initialization block. -/
def code : List Stmt := program.body.drop 3 |>.take 2

theorem code_source : code =
    [.assign [.name "_MR_BASES_64"] (.tuple (bases.map (fun a => .constant (.int a)))),
     .assign [.name "_SMALL_PRIMES"] (.tuple (smallPrimes.map (fun p => .constant (.int p))))] := rfl

def heap (caller : Mutable.Heap) : Mutable.Heap :=
  caller ++ [.tuple baseValues, .tuple smallValues]

def env (caller : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal (fun _ => none) "_MR_BASES_64" (.ref caller.length))
    "_SMALL_PRIMES" (.ref (caller.length + 1))

def globals (caller : Mutable.Heap) : Mutable.Globals :=
  [("_MR_BASES_64", .ref caller.length), ("_SMALL_PRIMES", .ref (caller.length + 1))]

/-- This executes both tuple displays and assignments, retaining an arbitrary
caller heap as a prefix. No table is assumed to have been initialized. -/
theorem executes (caller : Mutable.Heap) :
    BlockContinues program (fun _ => none) code caller (env caller) (heap caller) [] := by
  rw [code_source]
  have first : BlockContinues program (fun _ => none)
      [.assign [.name "_MR_BASES_64"] (.tuple (bases.map (fun a => .constant (.int a))))]
      caller (setLocal (fun _ => none) "_MR_BASES_64" (.ref caller.length))
      (caller ++ [.tuple baseValues]) [] := by
    apply BlockContinues.assign
    apply Evaluates.tuple
    · change Evaluates.Arguments program (fun _ => none) _ caller baseValues caller []
      repeat first | apply Evaluates.Arguments.cons Evaluates.constant | exact .nil
    · decide
  have second : BlockContinues program
      (setLocal (fun _ => none) "_MR_BASES_64" (.ref caller.length))
      [.assign [.name "_SMALL_PRIMES"] (.tuple (smallPrimes.map (fun p => .constant (.int p))))]
      (caller ++ [.tuple baseValues])
      (setLocal (setLocal (fun _ => none) "_MR_BASES_64" (.ref caller.length))
        "_SMALL_PRIMES" (.ref (caller ++ [Mutable.Object.tuple baseValues]).length))
      ((caller ++ [.tuple baseValues]) ++ [.tuple smallValues]) [] := by
    apply BlockContinues.assign
    apply Evaluates.tuple
    · repeat first | apply Evaluates.Arguments.cons Evaluates.constant | exact .nil
    · decide
  simpa [env, heap, List.append_assoc] using first.append second

/-- Exporting the two initialized names preserves the addresses established by
execution; it does not reconstruct either tuple from a copied list. -/
theorem exports (caller : Mutable.Heap) (name : String) :
    env caller name = (globals caller).lookup name := by
  by_cases small : name = "_SMALL_PRIMES"
  · subst name
    simp [env, globals, setLocal, List.lookup]
  · by_cases base : name = "_MR_BASES_64"
    · subst name
      simp [env, globals, setLocal]
    · simp [env, globals, setLocal, List.lookup, small, base,
        beq_eq_false_iff_ne.mpr small, beq_eq_false_iff_ne.mpr base]

/-- The raw helper's previously explicit table hypotheses now follow from the
heap and bindings produced by the actual source initialization block. -/
def context (caller : Mutable.Heap) : SourceGlobals (heap caller) (globals caller) where
  smallAddress := caller.length + 1
  basesAddress := caller.length
  smallBinding := by simp [globals, List.lookup]
  basesBinding := by simp [globals]
  smallTuple := by simp [heap]
  basesTuple := by simp [heap]

theorem caller_preserved (caller : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : caller[address]? = some object) : (heap caller)[address]? = some object := by
  have inside : address < caller.length := by
    by_contra outside
    simp [List.getElem?_eq_none (by omega : caller.length ≤ address)] at found
  simpa [heap, List.getElem?_append_left inside] using found

/-- Every signed input can use the executed setup state, with no public
arbitrary-base assumption and no restriction to the unsigned 64-bit range.
The conclusion is the exact algorithmic Miller–Rabin model, not Nat.Prime. -/
theorem result (n : Int) (caller : Mutable.Heap) :
    ReturnsValue program "is_prime" [.scalar (.int n)] (heap caller)
      (.scalar (.bool (isPrimeModel n.toNat))) (heap caller) (globals caller) :=
  raw_result n (heap caller) (globals caller) (context caller)

end solvers.«60».is_prime.initialization
