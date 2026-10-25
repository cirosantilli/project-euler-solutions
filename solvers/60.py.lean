import ProjectEulerSolutions.Lib.MillerRabin
import «.formalsw».solvers.«60.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.DecimalShift
import ProjectEulerSolutions.Lib.PrimeSieve
import ProjectEulerSolutions.Lib.IntPairCache
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.CliqueSearch
import lang.py.FormalSW.ComprehensionCalls
import Mathlib.Data.Nat.Sqrt

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


namespace solvers.«60».primes_up_to
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P60Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
variable {globals : Mutable.Globals}

open ProjectEulerSolutions.PrimeSieve
private def code : List Stmt := executableBody ((findFunctionIn program.body "primes_up_to").getD ([], [])).2
private def loopBody : List Stmt := match (code[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def markBody : List Stmt := match (loopBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body | _ => []
private def markEnv (env : HeapCalls.Env) (p : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "start" (number (p*p : Nat))) "step" (number p)

private theorem mark_result (env : HeapCalls.Env) (heap : Mutable.Heap) (address limit p : Nat)
    (bytes : List UInt8) (lower : 2 ≤ p) (inside : p*p ≤ limit) (size : bytes.length = limit+1)
    (getP : env "i" = some (number p)) (getLimit : env "limit" = some (number limit))
    (getBuffer : env "sieve" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv, BlockContinues (globals := globals) program env markBody heap finalEnv (heap.set address (.bytearray (mark bytes p))) ∧
      finalEnv "limit" = env "limit" ∧ finalEnv "sieve" = env "sieve" := by
  let e1 := setLocal env "start" (number (p*p : Nat))
  let e2 := markEnv env p
  let count : Int := ((limit : Int)-p*p)/p+1
  have first : BlockContinues (globals := globals) program env [markBody[0]'(by decide)] heap e1 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP)
      (by simp [Mutable.binary, scalarBinary])
  have second : BlockContinues (globals := globals) program e1 [markBody[1]'(by decide)] heap e2 heap :=
    BlockContinues.assign (Evaluates.lookup (by simpa [e1, setLocal] using getP))
  have third : BlockContinues (globals := globals) program e2 [markBody[2]'(by decide)] heap e2
      (heap.set address (.bytearray (mark bytes p))) := by
    apply BlockContinues.setSlice (value := .scalar (.bytes (List.replicate count.toNat 0)))
      (afterRhs := heap) (afterTarget := heap) (receiver := .ref address)
      (lowerValue := number (p*p : Nat)) (upperValue := number (limit+1 : Nat)) (strideValue := number p)
    · apply Evaluates.binary (av := .scalar (.bytes [0])) (bv := number count) (middle := heap) (afterRight := heap)
      · exact Evaluates.constant
      · apply Evaluates.binary (av := number (((limit : Int)-p*p)/p)) (bv := number 1)
          (middle := heap) (afterRight := heap)
        · apply Evaluates.binary (av := number ((limit : Int)-p*p)) (bv := number p)
            (middle := heap) (afterRight := heap)
          · exact Evaluates.binary (av := number limit) (bv := number (p*p : Nat))
              (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getLimit))
              (Evaluates.lookup (by simp [e2, markEnv, setLocal])) (by simp [Mutable.binary, scalarBinary, number])
          · exact Evaluates.lookup (by simp [e2, markEnv, setLocal])
          · simp [Mutable.binary, scalarBinary, number, show p ≠ 0 by omega, show 0 < p by omega]
        · exact Evaluates.constant
        · rfl
      · simp [Mutable.binary, scalarBinary]
    · exact .cons (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getBuffer))
        (.cons (Evaluates.lookup (by simp [e2, markEnv, setLocal]))
          (.cons (Evaluates.binary (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getLimit))
            Evaluates.constant (by simp [Mutable.binary, scalarBinary, number]))
          (.cons (Evaluates.lookup (by simp [e2, markEnv, setLocal])) .nil)))
    · apply Mutable.setSlice_buffer heap address bytes _ _ _ _ _
        { start := some (p*p), stop := some (limit+1), step := some p } found
      · change ((if some (p : Int) = some 0 then Mutable.error "ValueError"
          else (pure { start := some ((p : Int)*p), stop := some ((limit : Int)+1), step := some (p : Int) } : Mutable.Eval FormalSW.Py.Builtin.Slice.Slice)) heap) = _
        rw [if_neg (by simp; omega)]
        simp
      · have result := mark_slice bytes p lower (by omega)
        have equal : (bytes.length : Int)-p*p-1 = (limit : Int)-p*p := by rw [size]; push_cast; ring
        rw [equal] at result
        simpa only [size, count, Nat.cast_add, Nat.cast_one] using result
  exact ⟨e2, first.append (second.append third), by simp [e2, markEnv, setLocal], by simp [e2, markEnv, setLocal]⟩

private theorem heap_inside {heap : Mutable.Heap} {address : Nat}
    (found : heap[address]? = some (object : Mutable.Object)) : address < heap.length :=
  (List.getElem?_eq_some_iff.mp found).1

private theorem heap_set_same (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  apply List.ext_getElem?
  intro i
  by_cases same : i = address
  · subst i
    rw [List.getElem?_set_self (heap_inside found), found]
  · simp [List.getElem?_set_ne (Ne.symm same)]

private def Invariant (limit p : Nat) (bytes : List UInt8) : Prop :=
  Sieved limit p bytes ∧ bytes[0]? = some 0 ∧ bytes[1]? = some 0

private theorem invariant_mark (limit p : Nat) (bytes : List UInt8) (large : 1 ≤ limit)
    (lower : 2 ≤ p) (invariant : Invariant limit p bytes) (prime : Nat.Prime p) :
    Invariant limit (p + 1) (mark bytes p) := by
  refine ⟨sieved_mark limit p bytes invariant.1 prime, ?_, ?_⟩
  · rw [mark_get bytes p 0 (by omega) (by have := invariant.1.1; omega)]
    simpa [show ¬ p ^ 2 ≤ 0 by nlinarith] using invariant.2.1
  · rw [mark_get bytes p 1 (by omega) (by have := invariant.1.1; omega)]
    simpa [show ¬ p ^ 2 ≤ 1 by nlinarith] using invariant.2.2

private def Stable (before after : HeapCalls.Env) : Prop :=
  after "limit" = before "limit" ∧ after "sieve" = before "sieve"

private theorem read_buffer (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat)
    (bytes : List UInt8) (valid : p < bytes.length) (getP : env "i" = some (number p))
    (getArray : env "sieve" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    Evaluates (globals := globals) program env (.subscript (.name "sieve") (.name "i")) heap (number bytes[p].toNat) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getArray) (Evaluates.lookup getP)
  exact Mutable.getItem_buffer heap address p bytes found valid

private theorem iteration (limit p : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (large : 1 ≤ limit) (lower : 2 ≤ p) (inside : p * p ≤ limit)
    (getP : env "i" = some (number p)) (getLimit : env "limit" = some (number limit))
    (getArray : env "sieve" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes))
    (invariant : Invariant limit p bytes) :
    ∃ finalEnv finalBytes, BlockContinues (globals := globals) program env loopBody heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Invariant limit (p + 1) finalBytes ∧ Stable env finalEnv := by
  have size := invariant.1.1
  have valid : p < bytes.length := by nlinarith
  have live : bytes[p].toNat ≠ 0 ↔ Nat.Prime p :=
    (invariant.1.2 p bytes[p] lower (List.getElem?_eq_getElem valid)).trans (survives_current p lower)
  by_cases prime : Nat.Prime p
  · obtain ⟨finalEnv, marked, savedLimit, savedArray⟩ := mark_result env heap address limit p bytes
      lower inside size getP getLimit getArray found
    refine ⟨finalEnv, mark bytes p, ?_, invariant_mark limit p bytes large lower invariant prime, savedLimit, savedArray⟩
    apply BlockContinues.branch (flag := true) (middle := heap)
      (read_buffer heap env address p bytes valid getP getArray found)
      (by simp [Mutable.truth, scalarTruth, live.mpr prime]) marked
  · refine ⟨env, bytes, ?_, ⟨sieved_skip limit p bytes invariant.1 prime, invariant.2⟩, rfl, rfl⟩
    rw [heap_set_same heap address _ found]
    apply BlockContinues.branch (flag := false) (middle := heap)
      (read_buffer heap env address p bytes valid getP getArray found)
    · have zero : bytes[p].toNat = 0 := by tauto
      simp [Mutable.truth, scalarTruth, zero]
    · exact .nil

private theorem loop (limit first count : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (large : 1 ≤ limit) (lower : 2 ≤ first)
    (endpoint : first + count = Nat.sqrt limit + 1)
    (getLimit : env "limit" = some (number limit)) (getArray : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (invariant : Invariant limit first bytes) :
    ∃ finalEnv finalBytes, BlockContinues (globals := globals) program env
      [.rangeStepLoop "i" first (Nat.sqrt limit + 1) 1 loopBody []] heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Invariant limit (Nat.sqrt limit + 1) finalBytes ∧ Stable env finalEnv := by
  induction count generalizing first bytes env heap with
  | zero =>
      have same : first = Nat.sqrt limit + 1 := by omega
      refine ⟨env, bytes, ?_, by simpa only [same] using invariant, rfl, rfl⟩
      rw [heap_set_same heap address _ found]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "i" (number first)
      have inside : first * first ≤ limit := by
        have below : first ≤ Nat.sqrt limit := by omega
        exact (Nat.mul_self_le_mul_self below).trans (Nat.sqrt_le limit)
      obtain ⟨middleEnv, middleBytes, iterated, nextInvariant, stable⟩ :=
        iteration limit first bytes bound heap address large lower inside rfl
          (by simpa [bound, setLocal] using getLimit) (by simpa [bound, setLocal] using getArray) found invariant
      let middleHeap := heap.set address (.bytearray middleBytes)
      obtain ⟨finalEnv, finalBytes, executed, finalInvariant, saved⟩ := ih (first + 1) middleBytes middleEnv middleHeap
        (by omega) (by omega) (stable.1.trans (by simpa [bound, setLocal] using getLimit))
        (stable.2.trans (by simpa [bound, setLocal] using getArray))
        (List.getElem?_set_self (heap_inside found)) nextInvariant
      refine ⟨finalEnv, finalBytes, ?_, finalInvariant, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, middleHeap, List.set_set] using iterated.append executed
      · exact ⟨saved.1.trans (stable.1.trans (by simp [bound, setLocal])),
          saved.2.trans (stable.2.trans (by simp [bound, setLocal]))⟩


private theorem initial_slice (limit : Nat) (large : 1 ≤ limit) :
    FormalSW.Py.Builtin.ByteArray.setSlice (List.replicate (limit+1) 1)
      { start := some 0, stop := some 2 } [0,0] = .ok (initial limit) := by
  have normalized : FormalSW.Py.Builtin.Slice.indices
      { start := some 0, stop := some 2 } (limit+1) = .ok (0,2,1) := by
    simp only [FormalSW.Py.Builtin.Slice.indices, Option.getD_none]
    change Except.ok (max (0 : Int) (min 0 ((limit+1 : Nat) : Int)), max 0 (min 2 ((limit+1 : Nat) : Int)), (1 : Int)) = _
    simp
    omega
  simp only [FormalSW.Py.Builtin.ByteArray.setSlice, FormalSW.Py.Builtin.List.setSlice,
    List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte, List.length_replicate, normalized]
  change Except.ok ([0,0] ++ (List.replicate (limit+1) 1).drop 2) = _
  simp [initial, show limit+1-2 = limit-1 by omega]

private def initializedEnv (env : HeapCalls.Env) (address limit : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "sieve" (.ref address)) "r" (number (Nat.sqrt limit))
private def initialHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray (initial limit)]
private theorem setup (limit : Nat) (large : 1 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (number limit)) :
    BlockContinues (globals := globals) program env ((code.drop 1).take 3) heap
      (initializedEnv env (heap.length+1) limit) (initialHeap heap limit) := by
  let e1 := setLocal env "sieve" (.ref (heap.length+1))
  let allocated := heap ++ [.bytearray [1], .bytearray (List.replicate (limit+1) 1)]
  have first : BlockContinues (globals := globals) program env [code[1]'(by decide)] heap e1 allocated := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := .ref heap.length) (middle := heap ++ [.bytearray [1]])
      (bv := number (limit+1 : Nat)) (afterRight := heap ++ [.bytearray [1]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
    · simp [allocated, Mutable.binary, Mutable.allocateBuffer, Except.map,
        show (BinOp.mult == .mult) = true from rfl]
  have second : BlockContinues (globals := globals) program e1 [code[2]'(by decide)] allocated e1 (initialHeap heap limit) := by
    have found : allocated[heap.length+1]? = some (.bytearray (List.replicate (limit+1) 1)) := by simp [allocated]
    have write := Mutable.setSlice_buffer allocated (heap.length+1) (List.replicate (limit+1) 1) [0,0] (initial limit)
      (.scalar (.int 0)) (.scalar (.int 2)) (.scalar .none)
      { start := some 0, stop := some 2 } found rfl (initial_slice limit large)
    have same : allocated.set (heap.length+1) (.bytearray (initial limit)) = initialHeap heap limit := by
      simp [allocated, initialHeap]
    rw [same] at write
    exact BlockContinues.setSlice Evaluates.constant
      (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))) write
  have third : BlockContinues (globals := globals) program e1 [code[3]'(by decide)] (initialHeap heap limit)
      (initializedEnv env (heap.length+1) limit) (initialHeap heap limit) := by
    apply BlockContinues.assign
    apply Evaluates.library_call (.cons (Evaluates.lookup (by simpa [e1, setLocal] using getLimit)) .nil) rfl
    change scalarResult (some (Math.isqrtCall [.int limit])) (initialHeap heap limit) = _
    have root : Math.isqrt limit = Nat.sqrt limit := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
    simp [scalarResult, Math.isqrtCall, root, show ¬ (limit : Int) < 0 by omega]
  exact first.append (second.append third)


/-- The exact source prefix initializes and marks an inclusive primality table, preserving the input heap. -/
theorem prepare (limit : Nat) (large : 1 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (number limit)) :
    ∃ finalEnv bytes, BlockContinues (globals := globals) program env ((code.drop 1).take 4) heap finalEnv
        (heap ++ [.bytearray [1], .bytearray bytes]) ∧
      finalEnv "limit" = some (number limit) ∧ finalEnv "sieve" = some (.ref (heap.length + 1)) ∧
      bytes.length = limit + 1 ∧ (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  let ready := initializedEnv env (heap.length + 1) limit
  have initialSieved : Sieved limit 2 (initial limit) := by
    by_cases larger : 2 ≤ limit
    · exact sieved_initial limit larger
    · have same : limit = 1 := by omega
      subst limit
      refine ⟨by decide, ?_⟩
      intro i byte lower found
      have inside := (List.getElem?_eq_some_iff.mp found).1
      have length : (initial 1).length = 2 := by decide
      omega
  have initialInvariant : Invariant limit 2 (initial limit) :=
    ⟨initialSieved, by simp [initial], by simp [initial]⟩
  have readyLimit : ready "limit" = some (number limit) := by simpa [ready, initializedEnv, setLocal] using getLimit
  have rootPositive : 1 ≤ Nat.sqrt limit := by simpa using Nat.sqrt_le_sqrt large
  obtain ⟨finalEnv, bytes, executed, sieved, stable⟩ := loop limit 2 (Nat.sqrt limit - 1) (initial limit)
    ready (initialHeap heap limit) (heap.length + 1) large (by omega) (by omega) readyLimit
    (by simp [ready, initializedEnv, setLocal]) (by simp [initialHeap]) initialInvariant
  have loopRun : BlockContinues (globals := globals) program ready [code[4]'(by decide)] (initialHeap heap limit) finalEnv
      (heap ++ [.bytearray [1], .bytearray bytes]) := by
    apply BlockContinues.for_range (middle := initialHeap heap limit)
      (values := [number 2, number (Nat.sqrt limit + 1 : Nat)])
      (.cons Evaluates.constant (.cons (Evaluates.binary (av := number (Nat.sqrt limit)) (bv := number 1)
        (middle := initialHeap heap limit) (afterRight := initialHeap heap limit)
        (Evaluates.lookup (by simp [ready, initializedEnv, setLocal])) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) .nil)) rfl
    simpa [initialHeap] using executed
  refine ⟨finalEnv, bytes, (setup limit large env heap getLimit).append loopRun,
    stable.1.trans readyLimit, stable.2.trans (by simp [ready, initializedEnv, setLocal]), sieved.1.1, ?_⟩
  intro i byte found
  by_cases lower : 2 ≤ i
  · have inside := (List.getElem?_eq_some_iff.mp found).1
    exact (sieved.1.2 i byte lower found).trans
      (survives_finished limit (Nat.sqrt limit + 1) i lower (by have := sieved.1.1; omega) (Nat.lt_succ_sqrt' limit))
  · have cases : i = 0 ∨ i = 1 := by omega
    rcases cases with rfl | rfl
    · rw [sieved.2.1] at found
      cases found
      simp [Nat.not_prime_zero]
    · rw [sieved.2.2] at found
      cases found
      simp [Nat.not_prime_one]

/-- Increasing primes in the requested inclusive interval. -/
def primes (limit : Nat) : List Nat := (List.range' 2 (limit-1)).filter Nat.Prime
def numbers (values : List Nat) : List Mutable.Value := values.map (fun (n : Nat) => number n)
private def selected (start count : Nat) : List Nat := (List.range' start count).filter Nat.Prime
private abbrev collectBody (collector : Nat) : List Stmt :=
  [.ifThen (.subscript (.name "sieve") (.name "i")) [.comprehensionAppend collector (.name "i")] []]
private theorem collect_loop (count start limit : Nat) (length : start+count = limit+1) (lower : 2 ≤ start)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (address collector : Nat) (separate : address ≠ collector)
    (bytes : List UInt8) (before : List Nat)
    (getBuffer : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = limit+1)
    (correct : ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i))
    (collected : heap[collector]? = some (.list (numbers before))) :
    ∃ finalEnv, BlockContinues (globals := globals) program env
      [.rangeStepLoop "i" start (limit+1) 1 (collectBody collector) []] heap finalEnv
      (heap.set collector (.list (numbers (before ++ selected start count)))) := by
  induction count generalizing start heap env before with
  | zero =>
      have done : start = limit+1 := by omega
      refine ⟨env, ?_⟩
      simp only [selected, List.range'_zero, List.filter_nil, List.append_nil, heap_set_same heap collector _ collected]
      apply BlockContinues.range_done _ .nil
      simp [Calls.rangeContinues, done]
  | succ count ih =>
      have valid : start < bytes.length := by omega
      let bound := setLocal env "i" (number start)
      have getBuffer' : bound "sieve" = some (.ref address) := by simpa [bound, setLocal] using getBuffer
      have read : Evaluates (globals := globals) program bound (.subscript (.name "sieve") (.name "i")) heap
          (number bytes[start].toNat) heap := by
        apply Evaluates.subscript (middle := heap) (afterIndex := heap) (by intros; intro h; cases h)
          (Evaluates.lookup getBuffer') (Evaluates.lookup rfl)
        exact Mutable.getItem_buffer heap address start bytes found valid
      have live := correct start bytes[start] lower (List.getElem?_eq_getElem valid)
      by_cases prime : Nat.Prime start
      · let middle := heap.set collector (.list (numbers (before ++ [start])))
        have middleFound : middle[address]? = some (.bytearray bytes) := by
          simpa [middle, List.getElem?_set_ne (Ne.symm separate)] using found
        have middleCollected : middle[collector]? = some (.list (numbers (before ++ [start]))) :=
          List.getElem?_set_self (heap_inside collected)
        have emitted : BlockContinues (globals := globals) program bound (collectBody collector) heap bound middle := by
          apply BlockContinues.branch (flag := true) read
          · simp [Mutable.truth, scalarTruth, live.mpr prime]
          · simpa [middle, numbers] using BlockContinues.comprehension_append
              (Evaluates.lookup (globals := globals) (module := program) (heap := heap) (show bound "i" = some (number start) from rfl)) collected
        obtain ⟨finalEnv, executed⟩ := ih (start+1) (by omega) (by omega) middle bound (before ++ [start])
          getBuffer' middleFound middleCollected
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, collectBody,
          middle, List.set_set, selected, List.range'_succ, prime, List.append_assoc, bound, Nat.cast_add, Nat.cast_one]
          using emitted.append executed
      · have skipped : BlockContinues (globals := globals) program bound (collectBody collector) heap bound heap := by
          apply BlockContinues.branch (flag := false) read
          · have zero : bytes[start].toNat = 0 := by tauto
            simp [Mutable.truth, scalarTruth, zero]
          · exact .nil
        obtain ⟨finalEnv, executed⟩ := ih (start+1) (by omega) (by omega) heap bound before getBuffer' found collected
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, collectBody,
          selected, List.range'_succ, prime, bound, Nat.cast_add, Nat.cast_one] using skipped.append executed

private abbrev outputExpression : Expr :=
  .listComp (.name "i") [("i", .call (.name "range") [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))],
    [.subscript (.name "sieve") (.name "i")])]
private theorem collect (limit : Nat) (large : 1 ≤ limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (bytes : List UInt8)
    (getLimit : env "limit" = some (number limit)) (getBuffer : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = limit+1)
    (correct : ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Evaluates (globals := globals) program env outputExpression heap (.ref heap.length) (heap ++ [.list (numbers (primes limit))]) := by
  have bound := heap_inside found
  obtain ⟨finalEnv, collected⟩ := collect_loop (limit-1) 2 limit (by omega) (by omega)
    (heap ++ [.list []])
    (comprehensionEnv env [("i", .call (.name "range") [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))],
      [.subscript (.name "sieve") (.name "i")])]) address heap.length (by omega) bytes []
    (by simpa [comprehensionEnv] using getBuffer)
    (by rw [List.getElem?_append_left bound]; exact found) size correct (by simp [numbers])
  apply Evaluates.list_comp_range (values := [number 2, number (limit+1 : Nat)]) (middle := heap)
  · exact .cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup getLimit)
      Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil)
  · rfl
  · simpa [primes, selected, List.set_append, comprehensionFilters, comprehensionBody, collectBody] using collected




end solvers.«60».primes_up_to

namespace solvers.«60».primes_up_to
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
variable {globals : Mutable.Globals}

/-- The positive source path returns exactly the increasing prime list. The
seed bytearray and sieved buffer remain as private allocations before it. -/
theorem positive_result (limit : Nat) (large : 2 ≤ limit) (heap : Mutable.Heap) :
    ∃ bytes, ReturnsValue program "primes_up_to" [number limit] heap (.ref (heap.length + 2))
        (heap ++ [.bytearray [1], .bytearray bytes, .list (numbers (primes limit))]) globals ∧
      bytes.length = limit + 1 ∧
      (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  let env := functionEnv ((findFunctionIn program.body "primes_up_to").getD ([], [])).1
    ((findFunctionIn program.body "primes_up_to").getD ([], [])).2 [number limit] globals
  have input : env "limit" = some (number limit) := by cases globals <;> rfl
  obtain ⟨finalEnv, bytes, prepared, getLimit, getBuffer, size, correct⟩ :=
    prepare (globals := globals) limit (by omega) env heap input
  let middle := heap ++ [Mutable.Object.bytearray [1], .bytearray bytes]
  have found : middle[heap.length + 1]? = some (.bytearray bytes) := by simp [middle]
  have collected := collect (globals := globals) limit (by omega) middle finalEnv
    (heap.length + 1) bytes getLimit getBuffer found size
    (fun i byte _ found => correct i byte found)
  refine ⟨bytes, ?_, size, correct⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
  · exact Evaluates.compare (Evaluates.lookup input) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  · rfl
  · have finished := prepared.returns (BlockReturns.return_value (rest := []) collected)
    simpa only [middle, List.length_append, List.length_cons, List.length_nil, Nat.add_zero,
      List.append_assoc, List.singleton_append, Nat.add_assoc] using finished

/-- Bounds below two, including every negative bound, execute the early list
return without allocating either bytearray or calling isqrt. -/
theorem below_two_result (limit : Int) (small : limit < 2) (heap : Mutable.Heap) :
    ReturnsValue program "primes_up_to" [number limit] heap (.ref heap.length)
      (heap ++ [.list []]) globals := by
  let env := functionEnv ((findFunctionIn program.body "primes_up_to").getD ([], [])).1
    ((findFunctionIn program.body "primes_up_to").getD ([], [])).2 [number limit] globals
  have input : env "limit" = some (number limit) := by cases globals <;> rfl
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
  · exact Evaluates.compare (Evaluates.lookup input) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])
  · rfl
  · exact BlockReturns.return_value (Evaluates.list .nil)

/-- All signed limits return the genuine inclusive prime list. Private buffer
allocations are explicit, and the complete caller heap remains unchanged. -/
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ buffers : Mutable.Heap,
      buffers.length = (if limit < 2 then 0 else 2) ∧
      ReturnsValue program "primes_up_to" [number limit] heap
        (.ref (heap.length + buffers.length))
        (heap ++ buffers ++ [.list (numbers (primes limit.toNat))]) globals := by
  by_cases small : limit < 2
  · have primesEmpty : primes limit.toNat = [] := by
      have length : limit.toNat - 1 = 0 := by omega
      simp [primes, length]
    refine ⟨[], by simp [small], ?_⟩
    simpa only [primesEmpty, numbers, List.map_nil, List.append_nil, List.length_nil, Nat.add_zero] using
      below_two_result (globals := globals) limit small heap
  · have nonnegative : 0 ≤ limit := by omega
    obtain ⟨bytes, returned, _, _⟩ := positive_result (globals := globals) limit.toNat (by omega) heap
    refine ⟨[.bytearray [1], .bytearray bytes], by simp [small], ?_⟩
    simpa only [Int.toNat_of_nonneg nonnegative, List.length_cons, List.length_nil, Nat.zero_add,
      List.append_assoc, List.singleton_append] using returned

theorem mem_primes (limit p : Nat) : p ∈ primes limit ↔ Nat.Prime p ∧ p ≤ limit := by
  simp only [primes, List.mem_filter, List.mem_range'_1, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨lower, upper⟩, prime⟩
    exact ⟨prime, by omega⟩
  · rintro ⟨prime, upper⟩
    have lower := prime.two_le
    exact ⟨⟨lower, by omega⟩, prime⟩

theorem mem_primes_signed (limit : Int) (p : Nat) :
    p ∈ primes limit.toNat ↔ Nat.Prime p ∧ (p : Int) ≤ limit := by
  rw [mem_primes]
  constructor
  · rintro ⟨prime, upper⟩
    have lower := prime.two_le
    exact ⟨prime, by omega⟩
  · rintro ⟨prime, upper⟩
    exact ⟨prime, by omega⟩

theorem primes_increasing (limit : Nat) : (primes limit).Pairwise (· < ·) := by
  unfold primes
  exact List.Pairwise.filter _ (List.pairwise_lt_range' 1 (by decide))

theorem primes_nodup (limit : Nat) : (primes limit).Nodup :=
  (primes_increasing limit).imp (fun less => Nat.ne_of_lt less)

theorem «Returns» (limit : Int) :
    (HeapCalls.call program "primes_up_to" [number limit] []).Returns := by
  obtain ⟨_, _, returned⟩ := result (globals := []) limit []
  exact returned.returns

theorem «Supported» (limit : Int) :
    (HeapCalls.call program "primes_up_to" [number limit] []).Supported :=
  («Returns» limit).supported

end solvers.«60».primes_up_to

namespace solvers.«60».is_prime
open FormalSW.PyAst

/-- Later sieve/list/cache allocations retain the initialized table objects. -/
def SourceGlobals.append (context : SourceGlobals heap globals) (extra : Mutable.Heap) :
    SourceGlobals (heap ++ extra) globals where
  smallAddress := context.smallAddress
  basesAddress := context.basesAddress
  smallBinding := context.smallBinding
  basesBinding := context.basesBinding
  smallTuple := by
    rw [List.getElem?_append_left ((List.getElem?_eq_some_iff.mp context.smallTuple).1)]
    exact context.smallTuple
  basesTuple := by
    rw [List.getElem?_append_left ((List.getElem?_eq_some_iff.mp context.basesTuple).1)]
    exact context.basesTuple

end solvers.«60».is_prime


namespace solvers.«60».pair_ok
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev program := Solvers.P60Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private abbrev empty : Mutable.Value := .scalar .none

def cacheCell : Expr := .subscript (.name "__formalsw_cell_solve_cache_local") (.constant (.int 0))
def powerCell : Expr := .subscript (.name "__formalsw_cell_solve_pow10_local") (.constant (.int 0))
def keyExpression : Expr := .tuple [.name "p", .name "q"]
def getExpression : Expr := .call (.attribute cacheCell "get") [.name "key"]
def hitTest : Expr := .unaryOp .not (.builtinCall .identity [.name "v", .constant .none])
def divisibleTest : Expr :=
  .compare (.binOp (.binOp (.name "p") .add (.name "q")) .mod (.constant (.int 3)))
    [(.eq, .constant (.int 0))]
def concatExpression (left right : String) : Expr :=
  .binOp (.binOp (.name left) .mult (.subscript powerCell (.name right))) .add (.name right)
def write (expression : Expr) : Stmt := .assign [.subscript cacheCell (.name "key")] expression
def reject : List Stmt := [write (.constant (.bool false)), .return (.constant (.bool false))]
def tail : List Stmt :=
  [.assign [.name "key"] keyExpression,
   .assign [.name "v"] getExpression,
   .ifThen hitTest [.return (.name "v")] [],
   .ifThen divisibleTest reject [],
   .assign [.name "c1"] (concatExpression "p" "q"),
   .ifThen (.unaryOp .not (.call (.name "is_prime") [.name "c1"])) reject [],
   .assign [.name "c2"] (concatExpression "q" "p"),
   .assign [.name "ok"] (.call (.name "is_prime") [.name "c2"]),
   write (.name "ok"), .return (.name "ok")]
def code : List Stmt :=
  .ifThen (.compare (.name "p") [(.gt, .name "q")])
    [.assign [.tuple [.name "p", .name "q"]] (.tuple [.name "q", .name "p"])] [] :: tail

theorem code_source :
    executableBody ((findFunctionIn program.body "__formalsw_local_solve_pair_ok").getD ([], [])).2 = code := rfl

def concatenated (left right : Int) : Int := left * (DecimalShift.decimalShift right : Int) + right
def answer (left right : Int) : Bool :=
  if (left + right) % 3 = 0 then false
  else MillerRabin.isPrimeModel (concatenated left right).toNat &&
    MillerRabin.isPrimeModel (concatenated right left).toNat

theorem answer_comm (left right : Int) : answer left right = answer right left := by
  simp only [answer, Int.add_comm right left]
  split <;> simp [Bool.and_comm]

variable {heap : Mutable.Heap} {env : HeapCalls.Env} {entries : List IntPairCache.Entry}
  {left right : Int} {cacheCell powerCell address : Nat} {globals : Mutable.Globals}
  {values : List Mutable.Value}

private theorem found_append (found : heap[address]? = some (object : Mutable.Object))
    (extra : Mutable.Heap) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left ((List.getElem?_eq_some_iff.mp found).1)]
  exact found

structure Store (heap : Mutable.Heap) (entries : List IntPairCache.Entry) (cacheCell powerCell : Nat) where
  cacheAddress : Nat
  powerAddress : Nat
  powers : List (Constant × Mutable.Value)
  cacheCellFound : heap[cacheCell]? = some (.list [.ref cacheAddress])
  powerCellFound : heap[powerCell]? = some (.list [.ref powerAddress])
  cacheFound : heap[cacheAddress]? = some (IntPairCache.object entries)
  powerFound : heap[powerAddress]? = some (.dict powers)
  separate : cacheAddress ≠ powerAddress

def Store.append (store : Store heap entries cacheCell powerCell) (extra : Mutable.Heap) :
    Store (heap ++ extra) entries cacheCell powerCell where
  cacheAddress := store.cacheAddress
  powerAddress := store.powerAddress
  powers := store.powers
  cacheCellFound := found_append store.cacheCellFound extra
  powerCellFound := found_append store.powerCellFound extra
  cacheFound := found_append store.cacheFound extra
  powerFound := found_append store.powerFound extra
  separate := store.separate

theorem Store.cache_ne_list (store : Store heap entries cacheCell powerCell)
    (found : heap[address]? = some (.list values)) : store.cacheAddress ≠ address := by
  intro same
  have impossible := store.cacheFound
  rw [same, found] at impossible
  cases entries <;> cases impossible

theorem Store.cache_ne_tuple (store : Store heap entries cacheCell powerCell)
    (found : heap[address]? = some (.tuple values)) : store.cacheAddress ≠ address := by
  intro same
  have impossible := store.cacheFound
  rw [same, found] at impossible
  cases entries <;> cases impossible

def Store.update (store : Store heap entries cacheCell powerCell) (updated : List IntPairCache.Entry) :
    Store (heap.set store.cacheAddress (IntPairCache.object updated)) updated cacheCell powerCell where
  cacheAddress := store.cacheAddress
  powerAddress := store.powerAddress
  powers := store.powers
  cacheCellFound := by
    rw [List.getElem?_set_ne (store.cache_ne_list store.cacheCellFound)]
    exact store.cacheCellFound
  powerCellFound := by
    rw [List.getElem?_set_ne (store.cache_ne_list store.powerCellFound)]
    exact store.powerCellFound
  cacheFound := List.getElem?_set_self ((List.getElem?_eq_some_iff.mp store.cacheFound).1)
  powerFound := by
    rw [List.getElem?_set_ne store.separate]
    exact store.powerFound
  separate := store.separate

def Store.update_globals (store : Store heap entries cacheCell powerCell)
    (context : is_prime.SourceGlobals heap globals) (updated : List IntPairCache.Entry) :
    is_prime.SourceGlobals (heap.set store.cacheAddress (IntPairCache.object updated)) globals where
  smallAddress := context.smallAddress
  basesAddress := context.basesAddress
  smallBinding := context.smallBinding
  basesBinding := context.basesBinding
  smallTuple := by
    rw [List.getElem?_set_ne (store.cache_ne_tuple context.smallTuple)]
    exact context.smallTuple
  basesTuple := by
    rw [List.getElem?_set_ne (store.cache_ne_tuple context.basesTuple)]
    exact context.basesTuple

structure Environment (env : HeapCalls.Env) (left right : Int) (cacheCell powerCell : Nat) : Prop where
  leftValue : env "p" = some (number left)
  rightValue : env "q" = some (number right)
  cacheValue : env "__formalsw_cell_solve_cache_local" = some (.ref cacheCell)
  powerValue : env "__formalsw_cell_solve_pow10_local" = some (.ref powerCell)

theorem Environment.bind (environment : Environment env left right cacheCell powerCell)
    (name : String) (value : Mutable.Value) (allowed : name ∈ ["key", "v", "c1", "c2", "ok"]) :
    Environment (setLocal env name value) left right cacheCell powerCell := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at allowed
  rcases allowed with rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨by simpa [setLocal] using environment.leftValue,
      by simpa [setLocal] using environment.rightValue,
      by simpa [setLocal] using environment.cacheValue,
      by simpa [setLocal] using environment.powerValue⟩

private theorem read_cell (env : HeapCalls.Env) (heap : Mutable.Heap) (name : String) (cell target : Nat)
    (bound : env name = some (.ref cell)) (found : heap[cell]? = some (.list [.ref target]))
    (globals : Mutable.Globals) :
    Evaluates program env (.subscript (.name name) (.constant (.int 0))) heap (.ref target) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup bound) Evaluates.constant
  exact Mutable.getItem_list heap cell 0 [.ref target] found (by simp)

private theorem get_evaluates (env : HeapCalls.Env) (heap : Mutable.Heap) (left right : Int)
    (cacheCell powerCell tuple : Nat) (entries : List IntPairCache.Entry)
    (store : Store heap entries cacheCell powerCell)
    (environment : Environment env left right cacheCell powerCell)
    (getKey : env "key" = some (.ref tuple))
    (tupleFound : heap[tuple]? = some (.tuple [number left, number right]))
    (globals : Mutable.Globals) :
    Evaluates program env getExpression heap
      (((IntPairCache.lookup entries left right).map flag).getD empty) heap globals := by
  apply Evaluates.dictionary_get_bound (by intros name impossible; cases impossible)
    (read_cell env heap _ _ _ environment.cacheValue store.cacheCellFound globals)
  · cases entries <;> simp [bindMethod, store.cacheFound, IntPairCache.object]
  · exact .cons (Evaluates.lookup getKey) .nil
  · exact IntPairCache.get heap store.cacheAddress tuple entries left right store.cacheFound tupleFound

private theorem hit_evaluates (env : HeapCalls.Env) (heap : Mutable.Heap) (cached : Option Bool)
    (getV : env "v" = some (((cached.map flag).getD empty))) (globals : Mutable.Globals) :
    Evaluates program env hitTest heap (flag cached.isSome) heap globals := by
  cases cached with
  | none =>
      apply Evaluates.unary (value := flag true)
      · exact Evaluates.builtin (.cons (Evaluates.lookup getV) (.cons Evaluates.constant .nil)) rfl
      · rfl
  | some value =>
      apply Evaluates.unary (value := flag false)
      · apply Evaluates.builtin (.cons (Evaluates.lookup getV) (.cons Evaluates.constant .nil))
        cases value <;> rfl
      · rfl

private theorem divisible_evaluates (env : HeapCalls.Env) (heap : Mutable.Heap) (left right : Int)
    (environment : Environment env left right cacheCell powerCell) (globals : Mutable.Globals) :
    Evaluates program env divisibleTest heap (flag (decide ((left + right) % 3 = 0))) heap globals := by
  apply Evaluates.compare (av := number ((left + right) % 3)) (bv := number 0)
  · exact Evaluates.binary (Evaluates.binary (Evaluates.lookup environment.leftValue)
      (Evaluates.lookup environment.rightValue) rfl) Evaluates.constant rfl
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]

private theorem reject_result (env : HeapCalls.Env) (heap : Mutable.Heap) (left right : Int)
    (cacheCell powerCell tuple : Nat) (entries : List IntPairCache.Entry)
    (store : Store heap entries cacheCell powerCell)
    (environment : Environment env left right cacheCell powerCell)
    (getKey : env "key" = some (.ref tuple))
    (tupleFound : heap[tuple]? = some (.tuple [number left, number right]))
    (following : List Stmt) (globals : Mutable.Globals) :
    BlockReturns program env (reject ++ following) heap (flag false)
      (heap.set store.cacheAddress (IntPairCache.object (IntPairCache.insert entries left right tuple false))) globals := by
  apply BlockReturns.setItem (by intros; simp) Evaluates.constant
    (.cons (read_cell env heap _ _ _ environment.cacheValue store.cacheCellFound globals)
      (.cons (Evaluates.lookup getKey) .nil))
    (IntPairCache.set heap store.cacheAddress tuple entries left right false store.cacheFound tupleFound)
  exact BlockReturns.return_value Evaluates.constant

private theorem concat_evaluates (env : HeapCalls.Env) (heap : Mutable.Heap)
    (first second : String) (left right : Int) (powerCell powerAddress : Nat)
    (powers : List (Constant × Mutable.Value))
    (getFirst : env first = some (number left)) (getSecond : env second = some (number right))
    (getCell : env "__formalsw_cell_solve_pow10_local" = some (.ref powerCell))
    (cellFound : heap[powerCell]? = some (.list [.ref powerAddress]))
    (mapFound : heap[powerAddress]? = some (.dict powers))
    (powerFound : ScalarDict.lookup powers (.int right) = some (number (DecimalShift.decimalShift right)))
    (globals : Mutable.Globals) :
    Evaluates program env (concatExpression first second) heap (number (concatenated left right)) heap globals := by
  apply Evaluates.binary (av := number (left * (DecimalShift.decimalShift right : Int)))
    (bv := number right)
  · apply Evaluates.binary (av := number left) (bv := number (DecimalShift.decimalShift right))
    · exact Evaluates.lookup getFirst
    · apply Evaluates.subscript (by intros; simp)
        (read_cell env heap _ _ _ getCell cellFound globals) (Evaluates.lookup getSecond)
      exact Mutable.getItem_dictionary heap powerAddress powers mapFound (.int right) powerFound
    · rfl
  · exact Evaluates.lookup getSecond
  · rfl

private theorem after_lookup (env : HeapCalls.Env) (heap : Mutable.Heap) (left right : Int)
    (cacheCell powerCell tuple : Nat) (entries : List IntPairCache.Entry)
    (store : Store heap entries cacheCell powerCell)
    (environment : Environment env left right cacheCell powerCell)
    (getKey : env "key" = some (.ref tuple))
    (getV : env "v" = some ((((IntPairCache.lookup entries left right).map flag).getD empty)))
    (tupleFound : heap[tuple]? = some (.tuple [number left, number right]))
    (leftPower : ScalarDict.lookup store.powers (.int left) = some (number (DecimalShift.decimalShift left)))
    (rightPower : ScalarDict.lookup store.powers (.int right) = some (number (DecimalShift.decimalShift right)))
    (correct : IntPairCache.Correct answer entries)
    (globals : Mutable.Globals) (context : is_prime.SourceGlobals heap globals) :
    BlockReturns program env (tail.drop 2) heap (flag (answer left right))
      (if (IntPairCache.lookup entries left right).isSome then heap
       else heap.set store.cacheAddress (IntPairCache.object
         (IntPairCache.insert entries left right tuple (answer left right)))) globals := by
  cases cached : IntPairCache.lookup entries left right with
  | some value =>
      have valueEq : value = answer left right := IntPairCache.lookup_correct correct cached
      have getValue : env "v" = some (flag (answer left right)) := by simpa [cached, valueEq] using getV
      simp only [Option.isSome_some, ↓reduceIte]
      apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
        (by simpa only [Option.isSome_some] using (hit_evaluates env heap (some value)
          (by simpa only [cached] using getV) globals)) rfl
      exact BlockReturns.return_value (Evaluates.lookup getValue)
  | none =>
      simp only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
      apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
        (by simpa only [Option.isSome_none] using (hit_evaluates env heap none
          (by simpa only [cached] using getV) globals)) rfl
      by_cases divisible : (left + right) % 3 = 0
      · have answerEq : answer left right = false := by simp [answer, divisible]
        rw [answerEq]
        apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
          (by simpa only [divisible, decide_true] using divisible_evaluates env heap left right environment globals) rfl
        exact reject_result env heap left right cacheCell powerCell tuple entries store environment getKey tupleFound _ globals
      · apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
          (by simpa only [divisible, decide_false] using divisible_evaluates env heap left right environment globals) rfl
        let c1 := concatenated left right
        let first := setLocal env "c1" (number c1)
        have firstEnvironment : Environment first left right cacheCell powerCell :=
          environment.bind "c1" (number c1) (by simp)
        have firstKey : first "key" = some (.ref tuple) := by simpa [first, setLocal] using getKey
        have firstValue : first "c1" = some (number c1) := by simp [first, setLocal]
        have firstPrime : Evaluates program first (.call (.name "is_prime") [.name "c1"])
            heap (flag (MillerRabin.isPrimeModel c1.toNat)) heap globals :=
          Evaluates.call (.cons (Evaluates.lookup firstValue) .nil) (is_prime.raw_result c1 heap globals context)
        have firstTest : Evaluates program first (.unaryOp .not (.call (.name "is_prime") [.name "c1"]))
            heap (flag (!MillerRabin.isPrimeModel c1.toNat)) heap globals := by
          apply Evaluates.unary firstPrime
          cases MillerRabin.isPrimeModel c1.toNat <;> rfl
        apply BlockReturns.assign
          (concat_evaluates env heap "p" "q" left right powerCell store.powerAddress store.powers
            environment.leftValue environment.rightValue environment.powerValue store.powerCellFound
            store.powerFound rightPower globals)
        cases accepted : MillerRabin.isPrimeModel c1.toNat with
        | false =>
            have answerEq : answer left right = false := by simp [answer, divisible, c1, accepted]
            rw [answerEq]
            apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
              (by simpa only [accepted, Bool.not_false] using firstTest) rfl
            exact reject_result first heap left right cacheCell powerCell tuple entries store firstEnvironment firstKey tupleFound _ globals
        | true =>
            apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
              (by simpa only [accepted, Bool.not_true] using firstTest) rfl
            let c2 := concatenated right left
            let second := setLocal first "c2" (number c2)
            let finished := setLocal second "ok" (flag (MillerRabin.isPrimeModel c2.toNat))
            have secondEnvironment : Environment second left right cacheCell powerCell :=
              firstEnvironment.bind "c2" (number c2) (by simp)
            have finishedEnvironment : Environment finished left right cacheCell powerCell :=
              secondEnvironment.bind "ok" (flag (MillerRabin.isPrimeModel c2.toNat)) (by simp)
            have finishedKey : finished "key" = some (.ref tuple) := by
              simpa [finished, second, first, setLocal] using getKey
            have finishedValue : finished "ok" = some (flag (MillerRabin.isPrimeModel c2.toNat)) := by
              simp [finished, setLocal]
            have answerEq : answer left right = MillerRabin.isPrimeModel c2.toNat := by
              simp [answer, divisible, c1, c2, accepted]
            rw [answerEq]
            apply BlockReturns.assign
              (concat_evaluates first heap "q" "p" right left powerCell store.powerAddress store.powers
                firstEnvironment.rightValue firstEnvironment.leftValue firstEnvironment.powerValue
                store.powerCellFound store.powerFound leftPower globals)
            apply BlockReturns.assign
              (Evaluates.call (.cons (Evaluates.lookup (show second "c2" = some (number c2) by simp [second, setLocal])) .nil)
                (is_prime.raw_result c2 heap globals context))
            apply BlockReturns.setItem (by intros; simp) (Evaluates.lookup finishedValue)
              (.cons (read_cell finished heap _ _ _ finishedEnvironment.cacheValue store.cacheCellFound globals)
                (.cons (Evaluates.lookup finishedKey) .nil))
              (IntPairCache.set heap store.cacheAddress tuple entries left right
                (MillerRabin.isPrimeModel c2.toNat) store.cacheFound tupleFound)
            exact BlockReturns.return_value (Evaluates.lookup finishedValue)

def keyHeap (heap : Mutable.Heap) (left right : Int) : Mutable.Heap :=
  heap ++ [.tuple [number left, number right]]

def nextEntries (entries : List IntPairCache.Entry) (left right : Int) (tuple : Nat) : List IntPairCache.Entry :=
  if (IntPairCache.lookup entries left right).isSome then entries
  else IntPairCache.insert entries left right tuple (answer left right)

def nextHeap (heap : Mutable.Heap) (cacheAddress : Nat) (entries : List IntPairCache.Entry)
    (left right : Int) : Mutable.Heap :=
  let keyed := keyHeap heap left right
  if (IntPairCache.lookup entries left right).isSome then keyed
  else keyed.set cacheAddress (IntPairCache.object (nextEntries entries left right heap.length))

theorem tail_result (env : HeapCalls.Env) (heap : Mutable.Heap) (left right : Int)
    (cacheCell powerCell : Nat) (entries : List IntPairCache.Entry)
    (store : Store heap entries cacheCell powerCell)
    (environment : Environment env left right cacheCell powerCell)
    (leftPower : ScalarDict.lookup store.powers (.int left) = some (number (DecimalShift.decimalShift left)))
    (rightPower : ScalarDict.lookup store.powers (.int right) = some (number (DecimalShift.decimalShift right)))
    (correct : IntPairCache.Correct answer entries)
    (globals : Mutable.Globals) (context : is_prime.SourceGlobals heap globals) :
    BlockReturns program env tail heap (flag (answer left right))
      (nextHeap heap store.cacheAddress entries left right) globals := by
  let keyed := setLocal env "key" (.ref heap.length)
  let bound := setLocal keyed "v" (((IntPairCache.lookup entries left right).map flag).getD empty)
  have keyedEnvironment : Environment keyed left right cacheCell powerCell :=
    environment.bind "key" (.ref heap.length) (by simp)
  have boundEnvironment : Environment bound left right cacheCell powerCell :=
    keyedEnvironment.bind "v" _ (by simp)
  have keyedKey : keyed "key" = some (.ref heap.length) := by simp [keyed, setLocal]
  have boundKey : bound "key" = some (.ref heap.length) := by simpa [bound, setLocal] using keyedKey
  have boundValue : bound "v" = some (((IntPairCache.lookup entries left right).map flag).getD empty) := by
    simp [bound, setLocal]
  let appended := store.append [.tuple [number left, number right]]
  have keyFound : (keyHeap heap left right)[heap.length]? = some (.tuple [number left, number right]) := by
    simp [keyHeap]
  have rest := after_lookup bound (keyHeap heap left right) left right cacheCell powerCell heap.length entries
    appended boundEnvironment boundKey boundValue keyFound
    (by simpa only [appended, Store.append] using leftPower)
    (by simpa only [appended, Store.append] using rightPower) correct globals
    (context.append [.tuple [number left, number right]])
  have same : nextHeap heap store.cacheAddress entries left right =
      (if (IntPairCache.lookup entries left right).isSome then keyHeap heap left right
       else (keyHeap heap left right).set store.cacheAddress
         (IntPairCache.object (IntPairCache.insert entries left right heap.length (answer left right)))) := by
    unfold nextHeap nextEntries
    split <;> simp_all
  rw [same]
  apply BlockReturns.assign
    (Evaluates.tuple (.cons (Evaluates.lookup environment.leftValue)
      (.cons (Evaluates.lookup environment.rightValue) .nil)) (by simp))
  apply BlockReturns.assign
    (get_evaluates keyed (keyHeap heap left right) left right cacheCell powerCell heap.length entries
      appended keyedEnvironment keyedKey keyFound globals)
  simpa only [bound, appended, Store.append] using rest

private theorem nextHeap_found (found : heap[address]? = some (object : Mutable.Object))
    (cacheAddress : Nat) (different : cacheAddress ≠ address)
    (entries : List IntPairCache.Entry) (left right : Int) :
    (nextHeap heap cacheAddress entries left right)[address]? = some object := by
  unfold nextHeap
  split
  · exact found_append found _
  · rw [List.getElem?_set_ne different]
    exact found_append found _

def Store.after (store : Store heap entries cacheCell powerCell) (left right : Int) :
    Store (nextHeap heap store.cacheAddress entries left right)
      (nextEntries entries left right heap.length) cacheCell powerCell where
  cacheAddress := store.cacheAddress
  powerAddress := store.powerAddress
  powers := store.powers
  cacheCellFound := nextHeap_found store.cacheCellFound _ (store.cache_ne_list store.cacheCellFound) entries left right
  powerCellFound := nextHeap_found store.powerCellFound _ (store.cache_ne_list store.powerCellFound) entries left right
  cacheFound := by
    by_cases hit : (IntPairCache.lookup entries left right).isSome = true
    · simp only [nextHeap, nextEntries, if_pos hit]
      exact found_append store.cacheFound _
    · simp only [nextHeap, nextEntries, if_neg hit]
      apply List.getElem?_set_self
      have bound := (List.getElem?_eq_some_iff.mp store.cacheFound).1
      simp only [keyHeap, List.length_append, List.length_cons, List.length_nil]
      omega
  powerFound := nextHeap_found store.powerFound _ store.separate entries left right
  separate := store.separate

def Store.after_globals (store : Store heap entries cacheCell powerCell) (left right : Int)
    (context : is_prime.SourceGlobals heap globals) :
    is_prime.SourceGlobals (nextHeap heap store.cacheAddress entries left right) globals := by
  let appended := store.append [.tuple [number left, number right]]
  let ready := context.append [.tuple [number left, number right]]
  by_cases hit : (IntPairCache.lookup entries left right).isSome = true
  · simpa only [nextHeap, if_pos hit] using ready
  · simpa only [nextHeap, nextEntries, if_neg hit] using
      appended.update_globals ready (IntPairCache.insert entries left right heap.length (answer left right))

@[simp] theorem Store.after_cacheAddress (store : Store heap entries cacheCell powerCell) (left right : Int) :
    (store.after left right).cacheAddress = store.cacheAddress := rfl

@[simp] theorem Store.after_powerAddress (store : Store heap entries cacheCell powerCell) (left right : Int) :
    (store.after left right).powerAddress = store.powerAddress := rfl

@[simp] theorem Store.after_powers (store : Store heap entries cacheCell powerCell) (left right : Int) :
    (store.after left right).powers = store.powers := rfl

theorem nextHeap_preserves (found : heap[address]? = some (object : Mutable.Object))
    (cacheAddress : Nat) (different : cacheAddress ≠ address)
    (entries : List IntPairCache.Entry) (left right : Int) :
    (nextHeap heap cacheAddress entries left right)[address]? = some object :=
  nextHeap_found found cacheAddress different entries left right

theorem nextHeap_length (heap : Mutable.Heap) (cacheAddress : Nat) (entries : List IntPairCache.Entry)
    (left right : Int) : (nextHeap heap cacheAddress entries left right).length = heap.length + 1 := by
  unfold nextHeap
  split <;> simp [keyHeap]

theorem nextEntries_correct (correct : IntPairCache.Correct answer entries) (left right : Int) (tuple : Nat) :
    IntPairCache.Correct answer (nextEntries entries left right tuple) := by
  unfold nextEntries
  split
  · exact correct
  · exact IntPairCache.insert_correct correct rfl

def orderedHeap (heap : Mutable.Heap) (left right : Int) : Mutable.Heap :=
  if right < left then heap ++ [.tuple [number right, number left]] else heap

private theorem found_ordered (found : heap[address]? = some (object : Mutable.Object))
    (left right : Int) : (orderedHeap heap left right)[address]? = some object := by
  unfold orderedHeap
  split
  · exact found_append found _
  · exact found

def Store.ordered (store : Store heap entries cacheCell powerCell) (left right : Int) :
    Store (orderedHeap heap left right) entries cacheCell powerCell where
  cacheAddress := store.cacheAddress
  powerAddress := store.powerAddress
  powers := store.powers
  cacheCellFound := found_ordered store.cacheCellFound left right
  powerCellFound := found_ordered store.powerCellFound left right
  cacheFound := found_ordered store.cacheFound left right
  powerFound := found_ordered store.powerFound left right
  separate := store.separate

def orderedGlobals (context : is_prime.SourceGlobals heap globals) (left right : Int) :
    is_prime.SourceGlobals (orderedHeap heap left right) globals where
  smallAddress := context.smallAddress
  basesAddress := context.basesAddress
  smallBinding := context.smallBinding
  basesBinding := context.basesBinding
  smallTuple := found_ordered context.smallTuple left right
  basesTuple := found_ordered context.basesTuple left right

def resultEntries (heap : Mutable.Heap) (entries : List IntPairCache.Entry) (left right : Int) : List IntPairCache.Entry :=
  nextEntries entries (min left right) (max left right) (orderedHeap heap left right).length

def resultHeap (heap : Mutable.Heap) (cacheAddress : Nat) (entries : List IntPairCache.Entry)
    (left right : Int) : Mutable.Heap :=
  nextHeap (orderedHeap heap left right) cacheAddress entries (min left right) (max left right)

/-- The complete lowered source helper, including the tuple-unpacking swap,
fresh lookup tuple, cached False results, and each actual primality call. -/
theorem result (left right : Int) (heap : Mutable.Heap) (cacheCell powerCell : Nat)
    (entries : List IntPairCache.Entry) (store : Store heap entries cacheCell powerCell)
    (leftPower : ScalarDict.lookup store.powers (.int left) = some (number (DecimalShift.decimalShift left)))
    (rightPower : ScalarDict.lookup store.powers (.int right) = some (number (DecimalShift.decimalShift right)))
    (correct : IntPairCache.Correct answer entries)
    (globals : Mutable.Globals) (context : is_prime.SourceGlobals heap globals) :
    ReturnsValue program "__formalsw_local_solve_pair_ok"
      [number left, number right, .ref cacheCell, .ref powerCell] heap (flag (answer left right))
      (resultHeap heap store.cacheAddress entries left right) globals := by
  let params := ((findFunctionIn program.body "__formalsw_local_solve_pair_ok").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "__formalsw_local_solve_pair_ok").getD ([], [])).2
  let env := functionEnv params sourceBody [number left, number right, .ref cacheCell, .ref powerCell] globals
  have environment : Environment env left right cacheCell powerCell := by
    constructor <;> cases globals <;> rfl
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (executableBody sourceBody) heap (flag (answer left right))
    (resultHeap heap store.cacheAddress entries left right) globals
  rw [code_source]
  by_cases reversed : right < left
  · have lower : right ≤ left := by omega
    have finalHeap : resultHeap heap store.cacheAddress entries left right =
        nextHeap (heap ++ [.tuple [number right, number left]]) store.cacheAddress entries right left := by
      simp only [resultHeap, orderedHeap, if_pos reversed, min_eq_right lower, max_eq_left lower]
    rw [finalHeap, answer_comm left right]
    apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
      (Evaluates.compare (Evaluates.lookup environment.leftValue) (Evaluates.lookup environment.rightValue)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, reversed])) rfl
    let swapped := assignNames env ["p", "q"] [number right, number left]
    have swappedEnvironment : Environment swapped right left cacheCell powerCell :=
      ⟨rfl, rfl,
        by simpa [swapped, assignNames, setLocal] using environment.cacheValue,
        by simpa [swapped, assignNames, setLocal] using environment.powerValue⟩
    let appended := store.append [.tuple [number right, number left]]
    have rest := tail_result swapped (heap ++ [.tuple [number right, number left]]) right left
      cacheCell powerCell entries appended swappedEnvironment
      (by simpa only [appended, Store.append] using rightPower)
      (by simpa only [appended, Store.append] using leftPower) correct globals
      (context.append [.tuple [number right, number left]])
    apply BlockReturns.unpack (items := [number right, number left])
      (afterUnpack := heap ++ [.tuple [number right, number left]]) rfl
      (Evaluates.tuple (.cons (Evaluates.lookup environment.rightValue)
        (.cons (Evaluates.lookup environment.leftValue) .nil)) (by simp))
      (by simp [Mutable.sequence]) rfl
    simpa only [appended, Store.append] using rest
  · have lower : left ≤ right := by omega
    have finalHeap : resultHeap heap store.cacheAddress entries left right =
        nextHeap heap store.cacheAddress entries left right := by
      simp only [resultHeap, orderedHeap, if_neg reversed, min_eq_left lower, max_eq_right lower]
    rw [finalHeap]
    apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
      (Evaluates.compare (Evaluates.lookup environment.leftValue) (Evaluates.lookup environment.rightValue)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, reversed])) rfl
    exact tail_result env heap left right cacheCell powerCell entries store environment leftPower rightPower correct globals context

def Store.result (store : Store heap entries cacheCell powerCell) (left right : Int) :
    Store (resultHeap heap store.cacheAddress entries left right)
      (resultEntries heap entries left right) cacheCell powerCell where
  cacheAddress := store.cacheAddress
  powerAddress := store.powerAddress
  powers := store.powers
  cacheCellFound := by
    simpa only [resultHeap, Store.ordered] using
      ((store.ordered left right).after (min left right) (max left right)).cacheCellFound
  powerCellFound := by
    simpa only [resultHeap, Store.ordered] using
      ((store.ordered left right).after (min left right) (max left right)).powerCellFound
  cacheFound := by
    simpa only [resultHeap, resultEntries, Store.ordered] using
      ((store.ordered left right).after (min left right) (max left right)).cacheFound
  powerFound := by
    simpa only [resultHeap, Store.ordered] using
      ((store.ordered left right).after (min left right) (max left right)).powerFound
  separate := store.separate

def Store.result_globals (store : Store heap entries cacheCell powerCell) (left right : Int)
    (context : is_prime.SourceGlobals heap globals) :
    is_prime.SourceGlobals (resultHeap heap store.cacheAddress entries left right) globals := by
  simpa only [resultHeap, Store.ordered] using
    (store.ordered left right).after_globals (min left right) (max left right) (orderedGlobals context left right)

theorem resultEntries_correct (correct : IntPairCache.Correct answer entries) (heap : Mutable.Heap) (left right : Int) :
    IntPairCache.Correct answer (resultEntries heap entries left right) :=
  nextEntries_correct correct (min left right) (max left right) _

@[simp] theorem Store.result_cacheAddress (store : Store heap entries cacheCell powerCell) (left right : Int) :
    (store.result left right).cacheAddress = store.cacheAddress := rfl

@[simp] theorem Store.result_powerAddress (store : Store heap entries cacheCell powerCell) (left right : Int) :
    (store.result left right).powerAddress = store.powerAddress := rfl

@[simp] theorem Store.result_powers (store : Store heap entries cacheCell powerCell) (left right : Int) :
    (store.result left right).powers = store.powers := rfl

theorem resultHeap_preserves (found : heap[address]? = some (object : Mutable.Object))
    (cacheAddress : Nat) (different : cacheAddress ≠ address)
    (entries : List IntPairCache.Entry) (left right : Int) :
    (resultHeap heap cacheAddress entries left right)[address]? = some object :=
  nextHeap_preserves (found_ordered found left right) cacheAddress different entries _ _

theorem resultHeap_length (heap : Mutable.Heap) (cacheAddress : Nat) (entries : List IntPairCache.Entry)
    (left right : Int) : (resultHeap heap cacheAddress entries left right).length =
      heap.length + (if right < left then 2 else 1) := by
  rw [resultHeap, nextHeap_length]
  unfold orderedHeap
  split <;> simp [Nat.add_assoc]

def initialHeap (base : Mutable.Heap) (powers : List (Constant × Mutable.Value)) : Mutable.Heap :=
  base ++ [.dict powers, .dict [], .list [.ref base.length], .list [.ref (base.length + 1)]]

/-- The concrete state established by the owner's dictionary and capture-cell
allocations, before the first pair test. It also witnesses the empty-cache case. -/
def initialStore (base : Mutable.Heap) (powers : List (Constant × Mutable.Value)) :
    Store (initialHeap base powers) [] (base.length + 3) (base.length + 2) where
  cacheAddress := base.length + 1
  powerAddress := base.length
  powers := powers
  cacheCellFound := by simp [initialHeap]
  powerCellFound := by simp [initialHeap]
  cacheFound := by simp [initialHeap, IntPairCache.object]
  powerFound := by simp [initialHeap]
  separate := by omega

def captureCode : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 2).take 3

theorem capture_source : captureCode =
    [.annAssign (.name "pair_cache")
      (.subscript (.name "dict") (.tuple [.subscript (.name "tuple") (.tuple [.name "int", .name "int"]), .name "bool"]))
      (some (.dict [])),
     .assign [.name "__formalsw_cell_solve_pow10_local"] (.list [.name "pow10_map"]),
     .assign [.name "__formalsw_cell_solve_cache_local"] (.list [.name "pair_cache"])] := rfl

def captureEnv (env : HeapCalls.Env) (base : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "pair_cache" (.ref (base.length + 1)))
    "__formalsw_cell_solve_pow10_local" (.ref (base.length + 2)))
    "__formalsw_cell_solve_cache_local" (.ref (base.length + 3))

/-- Execute the actual owner statements that create the empty cache and both
shared cells. The already-built power dictionary keeps its original address. -/
theorem capture_initialization (env : HeapCalls.Env) (base : Mutable.Heap)
    (powers : List (Constant × Mutable.Value))
    (getPowers : env "pow10_map" = some (.ref base.length)) (globals : Mutable.Globals) :
    BlockContinues program env captureCode (base ++ [.dict powers])
      (captureEnv env base) (initialHeap base powers) globals := by
  let first := setLocal env "pair_cache" (.ref (base.length + 1))
  let second := setLocal first "__formalsw_cell_solve_pow10_local" (.ref (base.length + 2))
  have firstRun : BlockContinues program env [captureCode[0]'(by decide)]
      (base ++ [.dict powers]) first ((base ++ [.dict powers]) ++ [.dict []]) globals := by
    apply BlockContinues.annAssign
    convert (Evaluates.dictionary (module := program) (env := env)
      (heap := base ++ [.dict powers]) (globals := globals) .nil) using 1
    simp [List.length_append]
  have secondRun : BlockContinues program first [captureCode[1]'(by decide)]
      ((base ++ [.dict powers]) ++ [.dict []]) second
      (((base ++ [.dict powers]) ++ [.dict []]) ++ [.list [.ref base.length]]) globals := by
    apply BlockContinues.assign
    convert (Evaluates.list (module := program) (env := first)
      (heap := (base ++ [.dict powers]) ++ [.dict []]) (globals := globals)
      (.cons (Evaluates.lookup (show first "pow10_map" = some (.ref base.length) by
        simpa [first, setLocal] using getPowers)) .nil)) using 1
    simp [List.length_append, List.append_assoc]
  have thirdRun : BlockContinues program second [captureCode[2]'(by decide)]
      (((base ++ [.dict powers]) ++ [.dict []]) ++ [.list [.ref base.length]]) (captureEnv env base)
      ((((base ++ [.dict powers]) ++ [.dict []]) ++ [.list [.ref base.length]]) ++ [.list [.ref (base.length + 1)]]) globals := by
    apply BlockContinues.assign
    convert (Evaluates.list (module := program) (env := second)
      (heap := ((base ++ [.dict powers]) ++ [.dict []]) ++ [.list [.ref base.length]]) (globals := globals)
      (.cons (Evaluates.lookup (show second "pair_cache" = some (.ref (base.length + 1)) by simp [second, first, setLocal])) .nil)) using 1
    simp [List.length_append, List.append_assoc]
  simpa [capture_source, captureEnv, first, second, initialHeap, List.append_assoc] using
    firstRun.append (secondRun.append thirdRun)

end solvers.«60».pair_ok


namespace solvers.«60».solve_setup
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev program := Solvers.P60Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev numbers (values : List Nat) : List Mutable.Value := values.map fun (value : Nat) => number value
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
variable {globals : Mutable.Globals}

def powerExpression : Expr := .call (.name "pow10_for") [.name "p"]
def powerGenerators : List (Expr × Expr × List Expr) := [(.name "p", .name "primes", [])]
def powerComprehension : Expr := .dictComp (.name "p") powerExpression powerGenerators
def powerBody (address : Nat) : List Stmt := [.comprehensionStore address (.name "p") powerExpression]

theorem power_source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[1]? =
      some (.annAssign (.name "pow10_map")
        (.subscript (.name "dict") (.tuple [.name "int", .name "int"])) (some powerComprehension)) := rfl

private theorem lookup_insert_integer (entries : List (Constant × Mutable.Value)) (key query : Int) (value : Mutable.Value) :
    ScalarDict.lookup (ScalarDict.insert entries (.int key) value) (.int query) =
      if query = key then some value else ScalarDict.lookup entries (.int query) := by
  induction entries with
  | nil =>
      by_cases same : query = key
      · subst query
        simp [ScalarDict.insert, ScalarDict.lookup, scalarCompare, evalCmpOp]
      · simp [ScalarDict.insert, ScalarDict.lookup, scalarCompare, evalCmpOp, same, Ne.symm same]
  | cons entry rest ih =>
      obtain ⟨stored, previous⟩ := entry
      cases stored with
      | int n =>
          by_cases inserted : n = key <;> by_cases selected : n = query <;>
            by_cases same : query = key <;>
            simp_all [ScalarDict.insert, ScalarDict.lookup, scalarCompare, evalCmpOp]
      | bool b =>
          cases b <;>
            by_cases inserted : (0 : Int) = key <;> by_cases selected : (0 : Int) = query <;>
            by_cases insertedOne : (1 : Int) = key <;> by_cases selectedOne : (1 : Int) = query <;>
            by_cases same : query = key <;>
            simp_all [ScalarDict.insert, ScalarDict.lookup, scalarCompare, evalCmpOp]
      | string text | bytes values | none =>
          by_cases same : query = key <;>
            simp_all [ScalarDict.insert, ScalarDict.lookup, scalarCompare]

def powerUpdate (entries : List (Constant × Mutable.Value)) (value : Nat) : List (Constant × Mutable.Value) :=
  ScalarDict.insert entries (.int value) (number (DecimalShift.decimalShift value))

def powerEntries (values : List Nat) : List (Constant × Mutable.Value) := values.foldl powerUpdate []

private theorem fold_lookup (remaining : List Nat) (entries : List (Constant × Mutable.Value)) (query : Nat) :
    ScalarDict.lookup (remaining.foldl powerUpdate entries) (.int query) =
      if query ∈ remaining then some (number (DecimalShift.decimalShift query)) else ScalarDict.lookup entries (.int query) := by
  induction remaining generalizing entries with
  | nil => simp
  | cons value rest ih =>
      rw [List.foldl_cons, ih]
      by_cases later : query ∈ rest
      · simp [later]
      · by_cases same : query = value
        · simp [same, powerUpdate, ScalarDict.lookup_insert_self]
        · simp [later, same, powerUpdate, lookup_insert_integer]

theorem powerEntries_lookup (values : List Nat) (query : Nat) (member : query ∈ values) :
    ScalarDict.lookup (powerEntries values) (.int query) = some (number (DecimalShift.decimalShift query)) := by
  simp only [powerEntries, fold_lookup, if_pos member]

private theorem power_store (heap : Mutable.Heap) (entries : List (Constant × Mutable.Value))
    (env : HeapCalls.Env) (value : Nat) (getValue : env "p" = some (number value)) :
    BlockContinues program env (powerBody heap.length) (heap ++ [.dict entries]) env
      (heap ++ [.dict (powerUpdate entries value)]) globals := by
  have calculated : Evaluates program env powerExpression (heap ++ [.dict entries])
      (number (DecimalShift.decimalShift value)) (heap ++ [.dict entries]) globals :=
    Evaluates.call (.cons (Evaluates.lookup getValue) .nil) (pow10_for.result value _ globals)
  have inserted := Mutable.setItem_dictionary (heap ++ [.dict entries]) heap.length entries
    (by simp) (.int value) (number (DecimalShift.decimalShift value))
  apply BlockContinues.comprehension_store (.cons (Evaluates.lookup getValue) (.cons calculated .nil))
  simpa [powerUpdate, List.set_append_right] using inserted

private theorem power_loop (heap : Mutable.Heap) (address : Nat) (values remaining : List Nat)
    (found : heap[address]? = some (.list (numbers values))) (position : Nat)
    (suffix : values.drop position = remaining)
    (entries : List (Constant × Mutable.Value)) (env : HeapCalls.Env) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "p" (.ref address) position (powerBody heap.length) []]
      (heap ++ [.dict entries]) finalEnv
      (heap ++ [.dict (remaining.foldl powerUpdate entries)]) globals := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  have sequence (entries : List (Constant × Mutable.Value)) :
      Mutable.sequence (.ref address) (heap ++ [.dict entries]) =
        some (.ok (numbers values), heap ++ [.dict entries]) := by
    simp [Mutable.sequence, List.getElem?_append_left inside, found]
  induction remaining generalizing position entries env with
  | nil =>
      refine ⟨env, ?_⟩
      apply BlockContinues.sequence_done (items := numbers values)
        (by simpa only [sequenceValue] using sequence entries)
      · have missing : values[position]? = none := by
          simpa only [suffix, List.head?_nil] using (List.head?_drop (l := values) (i := position)).symm
        simp only [numbers, List.getElem?_map, missing, Option.map_none]
      · exact .nil
  | cons value rest ih =>
      let bound := setLocal env "p" (number value)
      have emitted := power_store (globals := globals) heap entries bound value (by simp [bound, setLocal])
      obtain ⟨finalEnv, continued⟩ := ih (position + 1)
        (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]) (powerUpdate entries value) bound
      refine ⟨finalEnv, ?_⟩
      have item : values[position]? = some value := by
        simpa only [suffix, List.head?_cons] using (List.head?_drop (l := values) (i := position)).symm
      apply BlockContinues.sequence_step (items := numbers values) (value := number value)
        (by simpa only [sequenceValue] using sequence entries)
        (by simp only [numbers, List.getElem?_map, item, Option.map_some])
      simpa only [Calls.loopContinuation, powerBody, Calls.loopControl, Calls.loopControlStmt,
        Bool.false_or, Bool.false_eq_true, if_false, List.foldl_cons] using emitted.append continued

/-- The source comprehension runs the actual decimal-shift helper for every
prime, allocates one dictionary, and preserves all existing heap objects. -/
theorem powers_evaluate (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat) (values : List Nat)
    (getPrimes : env "primes" = some (.ref address))
    (found : heap[address]? = some (.list (numbers values))) :
    Evaluates program env powerComprehension heap (.ref heap.length)
      (heap ++ [.dict (powerEntries values)]) globals := by
  let masked := comprehensionTargetsEnv env powerGenerators
  obtain ⟨finalEnv, executed⟩ := power_loop (globals := globals) heap address values values found 0 rfl [] masked
  apply Evaluates.dict_comp_sequence (receiver := .ref address) (items := numbers values)
    (middle := heap) (finalEnv := finalEnv)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (Evaluates.lookup getPrimes) (by simp [Mutable.sequence, found])
  simpa only [powerGenerators, comprehensionTargetName, comprehensionTargetBody, comprehensionFilters,
    List.foldr_nil, comprehensionTargetsBody, sequenceSource, powerEntries, masked, powerBody, powerExpression] using executed

def keep (value : Nat) : Bool := decide (value ≠ 2 ∧ value ≠ 5)
def literal : Mutable.Object := .tuple [number 2, number 5]
def filterExpression : Expr :=
  .membership (.name "p") (.tuple [.constant (.int 2), .constant (.int 5)]) true
def primeExpression : Expr :=
  .listComp (.name "p") [("p", .call (.name "primes_up_to") [.name "limit"], [filterExpression])]
def filterBody (collector : Nat) : List Stmt :=
  [.ifThen filterExpression [.comprehensionAppend collector (.name "p")] []]

theorem filter_source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[0]? =
      some (.assign [.name "primes"] primeExpression) := rfl

private theorem filter_evaluates (env : HeapCalls.Env) (heap : Mutable.Heap) (value : Nat)
    (getP : env "p" = some (number value)) :
    Evaluates program env filterExpression heap (flag (keep value)) (heap ++ [literal]) globals := by
  apply Evaluates.membership
    (.cons (Evaluates.lookup getP) (.cons (Evaluates.tuple
      (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp)) .nil))
  by_cases two : value = 2
  · subst value
    simp [membership, sequenceContains, literal, number, flag, keep, Except.map]
  · by_cases five : value = 5
    · subst value
      simp [membership, sequenceContains, HeapCalls.compare, Mutable.truth, scalarTruth,
        scalarCompare, evalCmpOp, literal, number, flag, keep, Except.map]
    · have notTwo : (2 : Int) ≠ value := by exact_mod_cast Ne.symm two
      have notFive : (5 : Int) ≠ value := by exact_mod_cast Ne.symm five
      simp [membership, sequenceContains, HeapCalls.compare, Mutable.truth, scalarTruth, scalarCompare,
        evalCmpOp, number, flag, literal, keep, notTwo, notFive, two, five, Except.map]

private theorem filter_loop (base : Mutable.Heap) (address : Nat) (values remaining : List Nat)
    (found : base[address]? = some (.list (numbers values))) (position : Nat)
    (suffix : values.drop position = remaining)
    (before : List Nat) (garbage : Mutable.Heap) (env : HeapCalls.Env) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "p" (.ref address) position (filterBody base.length) []]
      (HeapCollection.working base (numbers before) garbage) finalEnv
      (HeapCollection.working base (numbers (before ++ remaining.filter keep))
        (garbage ++ List.replicate remaining.length literal)) globals := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  have sequence (before : List Nat) (garbage : Mutable.Heap) :
      Mutable.sequence (.ref address) (HeapCollection.working base (numbers before) garbage) =
        some (.ok (numbers values), HeapCollection.working base (numbers before) garbage) := by
    simp [Mutable.sequence, HeapCollection.working, List.getElem?_append_left inside, found]
  induction remaining generalizing position before garbage env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.filter_nil, List.append_nil, List.length_nil, List.replicate_zero]
      apply BlockContinues.sequence_done (items := numbers values)
        (by simpa only [sequenceValue] using sequence before garbage)
      · have missing : values[position]? = none := by
          simpa only [suffix, List.head?_nil] using (List.head?_drop (l := values) (i := position)).symm
        simp only [numbers, List.getElem?_map, missing, Option.map_none]
      · exact .nil
  | cons value rest ih =>
      let bound := setLocal env "p" (number value)
      have tested := filter_evaluates (globals := globals) bound
        (HeapCollection.working base (numbers before) garbage) value (by simp [bound, setLocal])
      have item : values[position]? = some value := by
        simpa only [suffix, List.head?_cons] using (List.head?_drop (l := values) (i := position)).symm
      cases selected : keep value with
      | false =>
          have skipped : BlockContinues program bound (filterBody base.length)
              (HeapCollection.working base (numbers before) garbage) bound
              (HeapCollection.working base (numbers before) (garbage ++ [literal])) globals := by
            apply BlockContinues.branch (flag := false)
              (by simpa only [selected, HeapCollection.append] using tested) rfl
            exact .nil
          obtain ⟨finalEnv, continued⟩ := ih (position + 1)
            (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]) before (garbage ++ [literal]) bound
          refine ⟨finalEnv, ?_⟩
          apply BlockContinues.sequence_step (items := numbers values) (value := number value)
            (by simpa only [sequenceValue] using sequence before garbage)
            (by simp only [numbers, List.getElem?_map, item, Option.map_some])
          simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, filterBody,
            selected, List.replicate_succ, List.append_assoc] using skipped.append continued
      | true =>
          have emitted : BlockContinues program bound (filterBody base.length)
              (HeapCollection.working base (numbers before) garbage) bound
              (HeapCollection.working base (numbers (before ++ [value])) (garbage ++ [literal])) globals := by
            apply BlockContinues.branch (flag := true)
              (by simpa only [selected, HeapCollection.append] using tested) rfl
            simpa only [numbers, List.map_append, List.map_cons, List.map_nil, List.append_nil] using
              (HeapCollection.emit (program := program) base (numbers before) (garbage ++ [literal]) [] (number value)
                (by simpa only [List.append_nil] using (Evaluates.lookup (globals := globals) (module := program)
                  (heap := HeapCollection.working base (numbers before) (garbage ++ [literal]))
                  (show bound "p" = some (number value) by simp [bound, setLocal]))))
          obtain ⟨finalEnv, continued⟩ := ih (position + 1)
            (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]) (before ++ [value]) (garbage ++ [literal]) bound
          refine ⟨finalEnv, ?_⟩
          apply BlockContinues.sequence_step (items := numbers values) (value := number value)
            (by simpa only [sequenceValue] using sequence before garbage)
            (by simp only [numbers, List.getElem?_map, item, Option.map_some])
          simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, filterBody,
            selected, List.replicate_succ, List.append_assoc] using emitted.append continued

/-- The live input sequence remains in the heap prefix. Each source tuple
display contributes one temporary object, including when its prime is rejected. -/
theorem filter_evaluate (env : HeapCalls.Env) (heap middle : Mutable.Heap) (iterable : Expr)
    (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (address : Nat) (values : List Nat)
    (evaluated : Evaluates program env iterable heap (.ref address) middle globals)
    (found : middle[address]? = some (.list (numbers values))) :
    Evaluates program env (.listComp (.name "p") [("p", iterable, [filterExpression])])
      heap (.ref middle.length)
      (HeapCollection.working middle (numbers (values.filter keep)) (List.replicate values.length literal)) globals := by
  let masked := comprehensionEnv env [("p", iterable, [filterExpression])]
  obtain ⟨finalEnv, executed⟩ := filter_loop (globals := globals) middle address values values found 0 rfl [] [] masked
  apply Evaluates.list_comp_sequence (items := numbers values) (finalEnv := finalEnv) ordinary evaluated
    (by simp [Mutable.sequence, found])
  simpa only [masked, comprehensionFilters, comprehensionBody, List.nil_append,
    filterBody, numbers, List.map_nil, HeapCollection.working] using executed

def vertices (limit : Int) : List Nat := (primes_up_to.primes limit.toNat).filter keep

def sieveHeap (heap buffers : Mutable.Heap) (limit : Int) : Mutable.Heap :=
  heap ++ buffers ++ [.list (numbers (primes_up_to.primes limit.toNat))]

def filteredHeap (heap buffers : Mutable.Heap) (limit : Int) : Mutable.Heap :=
  HeapCollection.working (sieveHeap heap buffers limit) (numbers (vertices limit))
    (List.replicate (primes_up_to.primes limit.toNat).length literal)

def prefixHeap (heap buffers : Mutable.Heap) (limit : Int) : Mutable.Heap :=
  pair_ok.initialHeap (filteredHeap heap buffers limit) (powerEntries (vertices limit))

def prefixEnv (env : HeapCalls.Env) (heap buffers : Mutable.Heap) (limit : Int) : HeapCalls.Env :=
  pair_ok.captureEnv
    (setLocal (setLocal env "primes" (.ref (sieveHeap heap buffers limit).length))
      "pow10_map" (.ref (filteredHeap heap buffers limit).length)) (filteredHeap heap buffers limit)

def prefixCode : List Stmt :=
  (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).take 5

theorem prefix_source : prefixCode =
    [.assign [.name "primes"] primeExpression,
     .annAssign (.name "pow10_map") (.subscript (.name "dict") (.tuple [.name "int", .name "int"]))
       (some powerComprehension)] ++ pair_ok.captureCode := rfl

/-- The actual first five owner statements initialize the candidate primes,
power dictionary, empty pair cache, and shared cells for every signed limit. -/
theorem prefix_result (env : HeapCalls.Env) (heap : Mutable.Heap) (limit : Int)
    (getLimit : env "limit" = some (number limit)) :
    ∃ buffers : Mutable.Heap,
      buffers.length = (if limit < 2 then 0 else 2) ∧
      BlockContinues program env prefixCode heap (prefixEnv env heap buffers limit)
        (prefixHeap heap buffers limit) globals := by
  obtain ⟨buffers, size, returned⟩ := primes_up_to.result (globals := globals) limit heap
  let sieved := sieveHeap heap buffers limit
  let filtered := filteredHeap heap buffers limit
  let ready := setLocal env "primes" (.ref sieved.length)
  let mapped := setLocal ready "pow10_map" (.ref filtered.length)
  have found : sieved[(heap ++ buffers).length]? =
      some (.list (numbers (primes_up_to.primes limit.toNat))) := by simp [sieved, sieveHeap]
  have call : Evaluates program env (.call (.name "primes_up_to") [.name "limit"]) heap
      (.ref (heap ++ buffers).length) sieved globals := by
    apply Evaluates.call (.cons (Evaluates.lookup getLimit) .nil)
    simpa only [List.length_append] using returned
  have primeEval : Evaluates program env primeExpression heap (.ref sieved.length) filtered globals :=
    filter_evaluate env heap sieved (.call (.name "primes_up_to") [.name "limit"])
      (by intros; simp) (heap ++ buffers).length _ call found
  have first : BlockContinues program env [.assign [.name "primes"] primeExpression]
      heap ready filtered globals := BlockContinues.assign primeEval
  have filteredFound : filtered[sieved.length]? = some (.list (numbers (vertices limit))) := by
    exact HeapCollection.found sieved _ _
  have mappedEval := powers_evaluate (globals := globals) ready filtered sieved.length (vertices limit)
    (by simp [ready, setLocal]) filteredFound
  have second : BlockContinues program ready
      [.annAssign (.name "pow10_map") (.subscript (.name "dict") (.tuple [.name "int", .name "int"]))
        (some powerComprehension)] filtered mapped (filtered ++ [.dict (powerEntries (vertices limit))]) globals :=
    BlockContinues.annAssign mappedEval
  have captured := pair_ok.capture_initialization mapped filtered (powerEntries (vertices limit))
    (by simp [mapped, setLocal]) globals
  refine ⟨buffers, size, ?_⟩
  rw [prefix_source]
  exact first.append (second.append captured)

theorem mem_vertices (limit : Int) (p : Nat) : p ∈ vertices limit ↔
    Nat.Prime p ∧ 3 ≤ p ∧ (p : Int) ≤ limit ∧ p ≠ 5 := by
  simp only [vertices, List.mem_filter, primes_up_to.mem_primes_signed, keep, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨prime, upper⟩, notTwo, notFive⟩
    have lower := prime.two_le
    exact ⟨prime, by omega, upper, notFive⟩
  · rintro ⟨prime, lower, upper, notFive⟩
    exact ⟨⟨prime, upper⟩, by omega, notFive⟩

theorem vertices_increasing (limit : Int) : (vertices limit).Pairwise (· < ·) := by
  unfold vertices
  exact List.Pairwise.filter _ (primes_up_to.primes_increasing limit.toNat)

theorem vertices_nodup (limit : Int) : (vertices limit).Nodup :=
  (vertices_increasing limit).imp (fun less => Nat.ne_of_lt less)

def preparedStore (heap buffers : Mutable.Heap) (limit : Int) :
    pair_ok.Store (prefixHeap heap buffers limit) []
      ((filteredHeap heap buffers limit).length + 3) ((filteredHeap heap buffers limit).length + 2) :=
  pair_ok.initialStore (filteredHeap heap buffers limit) (powerEntries (vertices limit))

theorem preparedStore_power (heap buffers : Mutable.Heap) (limit : Int) (p : Nat) (member : p ∈ vertices limit) :
    ScalarDict.lookup (preparedStore heap buffers limit).powers (.int p) =
      some (number (DecimalShift.decimalShift p)) :=
  powerEntries_lookup (vertices limit) p member

theorem preparedStore_correct : IntPairCache.Correct pair_ok.answer [] :=
  IntPairCache.correct_nil pair_ok.answer

theorem prefix_primes_binding (env : HeapCalls.Env) (heap buffers : Mutable.Heap) (limit : Int) :
    prefixEnv env heap buffers limit "primes" = some (.ref (sieveHeap heap buffers limit).length) := by
  simp [prefixEnv, pair_ok.captureEnv, setLocal]

theorem prefix_primes_found (heap buffers : Mutable.Heap) (limit : Int) :
    (prefixHeap heap buffers limit)[(sieveHeap heap buffers limit).length]? =
      some (.list (numbers (vertices limit))) := by
  apply HeapCollection.prefix_found (filteredHeap heap buffers limit) _ _ _
  exact HeapCollection.found (sieveHeap heap buffers limit) _ _

theorem prefix_preserves (heap buffers : Mutable.Heap) (limit : Int) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (prefixHeap heap buffers limit)[address]? = some object := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  simp only [prefixHeap, pair_ok.initialHeap, filteredHeap, sieveHeap, HeapCollection.working, List.append_assoc]
  rw [List.getElem?_append_left inside]
  exact found

def prefix_globals (context : is_prime.SourceGlobals heap globals) (buffers : Mutable.Heap) (limit : Int) :
    is_prime.SourceGlobals (prefixHeap heap buffers limit) globals where
  smallAddress := context.smallAddress
  basesAddress := context.basesAddress
  smallBinding := context.smallBinding
  basesBinding := context.basesBinding
  smallTuple := prefix_preserves heap buffers limit _ _ context.smallTuple
  basesTuple := prefix_preserves heap buffers limit _ _ context.basesTuple

def sentinel : Int := 10 ^ 30

def searchHeap (base : Mutable.Heap) : Mutable.Heap :=
  base ++ [.list [number sentinel], .list [], .list [.ref (base.length + 1)]]

def bestEnv (env : HeapCalls.Env) (base : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal env "__formalsw_cell_solve_best_sum" (.ref base.length))
    "__formalsw_cell_solve_best_set" (.ref (base.length + 2))

def searchEnv (env : HeapCalls.Env) (base : Mutable.Heap) (values : List Nat) : HeapCalls.Env :=
  setLocal (bestEnv env base) "n" (number values.length)

def searchSetupCode : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 5).take 3

theorem searchSetup_source : searchSetupCode =
    [.assign [.name "__formalsw_cell_solve_best_sum"]
      (.list [.binOp (.constant (.int 10)) .pow (.constant (.int 30))]),
     .assign [.name "__formalsw_cell_solve_best_set"] (.list [.list []]),
     .assign [.name "n"] (.builtinCall .len [.name "primes"])] := rfl

theorem searchSetup_result (env : HeapCalls.Env) (base : Mutable.Heap) (address : Nat) (values : List Nat)
    (getPrimes : env "primes" = some (.ref address))
    (found : base[address]? = some (.list (numbers values))) :
    BlockContinues program env searchSetupCode base (searchEnv env base values) (searchHeap base) globals := by
  let first := setLocal env "__formalsw_cell_solve_best_sum" (.ref base.length)
  let ready := bestEnv env base
  have firstRun : BlockContinues program env [searchSetupCode[0]'(by decide)] base first
      (base ++ [.list [number sentinel]]) globals := by
    apply BlockContinues.assign
    exact Evaluates.list (.cons (Evaluates.binary Evaluates.constant Evaluates.constant rfl) .nil)
  have secondRun : BlockContinues program first [searchSetupCode[1]'(by decide)]
      (base ++ [.list [number sentinel]]) ready (searchHeap base) globals := by
    apply BlockContinues.assign
    convert (Evaluates.list (module := program) (env := first) (heap := base ++ [.list [number sentinel]]) (globals := globals)
      (.cons (Evaluates.list .nil) .nil)) using 1 <;>
      simp [searchHeap, List.length_append, List.append_assoc]
  have searchFound : (searchHeap base)[address]? = some (.list (numbers values)) :=
    HeapCollection.prefix_found base _ address _ found
  have lengthEval : Evaluates program ready (.builtinCall .len [.name "primes"])
      (searchHeap base) (number values.length) (searchHeap base) globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup (by simpa [ready, bestEnv, setLocal] using getPrimes)) .nil)
    simp [primitive, Mutable.length, searchFound, numbers]
  have thirdRun : BlockContinues program ready [searchSetupCode[2]'(by decide)]
      (searchHeap base) (searchEnv env base values) (searchHeap base) globals :=
    BlockContinues.assign lengthEval
  simpa [searchSetup_source] using firstRun.append (secondRun.append thirdRun)

def initializedHeap (heap buffers : Mutable.Heap) (limit : Int) : Mutable.Heap :=
  searchHeap (prefixHeap heap buffers limit)

def initializedEnv (env : HeapCalls.Env) (heap buffers : Mutable.Heap) (limit : Int) : HeapCalls.Env :=
  searchEnv (prefixEnv env heap buffers limit) (prefixHeap heap buffers limit) (vertices limit)

def initializationCode : List Stmt :=
  (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).take 8

theorem initialization_source : initializationCode = prefixCode ++ searchSetupCode := rfl

/-- Complete execution up to the first outer search iteration. No restriction
on the signed limit is introduced by the heap or cache invariants. -/
theorem initialization_result (env : HeapCalls.Env) (heap : Mutable.Heap) (limit : Int)
    (getLimit : env "limit" = some (number limit)) :
    ∃ buffers : Mutable.Heap,
      buffers.length = (if limit < 2 then 0 else 2) ∧
      BlockContinues program env initializationCode heap (initializedEnv env heap buffers limit)
        (initializedHeap heap buffers limit) globals := by
  obtain ⟨buffers, size, prefixRun⟩ := prefix_result (globals := globals) env heap limit getLimit
  refine ⟨buffers, size, ?_⟩
  rw [initialization_source]
  exact prefixRun.append (searchSetup_result (prefixEnv env heap buffers limit) (prefixHeap heap buffers limit)
    (sieveHeap heap buffers limit).length (vertices limit)
    (prefix_primes_binding env heap buffers limit) (prefix_primes_found heap buffers limit))

def initializedStore (heap buffers : Mutable.Heap) (limit : Int) :
    pair_ok.Store (initializedHeap heap buffers limit) []
      ((filteredHeap heap buffers limit).length + 3) ((filteredHeap heap buffers limit).length + 2) :=
  (preparedStore heap buffers limit).append
    [.list [number sentinel], .list [], .list [.ref ((prefixHeap heap buffers limit).length + 1)]]

theorem initializedStore_power (heap buffers : Mutable.Heap) (limit : Int) (p : Nat) (member : p ∈ vertices limit) :
    ScalarDict.lookup (initializedStore heap buffers limit).powers (.int p) =
      some (number (DecimalShift.decimalShift p)) :=
  preparedStore_power heap buffers limit p member

def initialized_globals (context : is_prime.SourceGlobals heap globals) (buffers : Mutable.Heap) (limit : Int) :
    is_prime.SourceGlobals (initializedHeap heap buffers limit) globals :=
  (prefix_globals context buffers limit).append
    [.list [number sentinel], .list [], .list [.ref ((prefixHeap heap buffers limit).length + 1)]]

theorem initialized_sum_binding (env : HeapCalls.Env) (heap buffers : Mutable.Heap) (limit : Int) :
    initializedEnv env heap buffers limit "__formalsw_cell_solve_best_sum" =
      some (.ref (prefixHeap heap buffers limit).length) := by
  simp [initializedEnv, searchEnv, bestEnv, setLocal]

theorem initialized_set_binding (env : HeapCalls.Env) (heap buffers : Mutable.Heap) (limit : Int) :
    initializedEnv env heap buffers limit "__formalsw_cell_solve_best_set" =
      some (.ref ((prefixHeap heap buffers limit).length + 2)) := by
  simp [initializedEnv, searchEnv, bestEnv, setLocal]

theorem initialized_count (env : HeapCalls.Env) (heap buffers : Mutable.Heap) (limit : Int) :
    initializedEnv env heap buffers limit "n" = some (number (vertices limit).length) := by
  simp [initializedEnv, searchEnv, setLocal]

end solvers.«60».solve_setup

namespace solvers.«60»
open FormalSW.PyAst
open ProjectEulerSolutions
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

/-- Physical state of the shared best-sum and best-set cells. The search proof
will separately relate their contents to the examined cliques. -/
structure Incumbent (heap : Mutable.Heap) (sumCell setCell : Nat) (bound : Int) (selected : List Nat) where
  selectedAddress : Nat
  sumFound : heap[sumCell]? = some (.list [number bound])
  setFound : heap[setCell]? = some (.list [.ref selectedAddress])
  selectedFound : heap[selectedAddress]? = some (.list (primes_up_to.numbers selected))

variable {heap : Mutable.Heap} {sumCell setCell : Nat} {bound : Int} {selected : List Nat}

def Incumbent.append (incumbent : Incumbent heap sumCell setCell bound selected) (extra : Mutable.Heap) :
    Incumbent (heap ++ extra) sumCell setCell bound selected where
  selectedAddress := incumbent.selectedAddress
  sumFound := HeapCollection.prefix_found heap extra _ _ incumbent.sumFound
  setFound := HeapCollection.prefix_found heap extra _ _ incumbent.setFound
  selectedFound := HeapCollection.prefix_found heap extra _ _ incumbent.selectedFound

/-- Cache activity retains the complete incumbent graph, including the list
address held in the best-set cell, across ordered or reversed pair calls. -/
def Incumbent.after_pair (incumbent : Incumbent heap sumCell setCell bound selected)
    (cacheCell powerCell : Nat) (entries : List IntPairCache.Entry)
    (store : pair_ok.Store heap entries cacheCell powerCell) (left right : Int) :
    Incumbent (pair_ok.resultHeap heap store.cacheAddress entries left right) sumCell setCell bound selected where
  selectedAddress := incumbent.selectedAddress
  sumFound := pair_ok.resultHeap_preserves incumbent.sumFound _ (store.cache_ne_list incumbent.sumFound) entries left right
  setFound := pair_ok.resultHeap_preserves incumbent.setFound _ (store.cache_ne_list incumbent.setFound) entries left right
  selectedFound := pair_ok.resultHeap_preserves incumbent.selectedFound _ (store.cache_ne_list incumbent.selectedFound) entries left right

def initializedIncumbent (heap buffers : Mutable.Heap) (limit : Int) :
    Incumbent (solve_setup.initializedHeap heap buffers limit)
      (solve_setup.prefixHeap heap buffers limit).length
      ((solve_setup.prefixHeap heap buffers limit).length + 2) solve_setup.sentinel [] where
  selectedAddress := (solve_setup.prefixHeap heap buffers limit).length + 1
  sumFound := by simp [solve_setup.initializedHeap, solve_setup.searchHeap, number]
  setFound := by simp [solve_setup.initializedHeap, solve_setup.searchHeap]
  selectedFound := by simp [solve_setup.initializedHeap, solve_setup.searchHeap, primes_up_to.numbers]

end solvers.«60»


namespace solvers.«60».dfs
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev program := Solvers.P60Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev numbers := primes_up_to.numbers
private abbrev nothing : Mutable.Value := .scalar .none
variable {heap : Mutable.Heap} {sumCell setCell : Nat} {bound : Int} {selected : List Nat}
  {globals : Mutable.Globals}

def sumExpression : Expr := .subscript (.name "__formalsw_cell_solve_best_sum") (.constant (.int 0))
def copyExpression : Expr := .subscript (.name "clique") (.slice none none none)
def improveTest : Expr := .compare (.name "curr_sum") [(.lt, sumExpression)]
def improveCode : List Stmt :=
  [.assign [sumExpression] (.name "curr_sum"),
   .assign [.subscript (.name "__formalsw_cell_solve_best_set") (.constant (.int 0))] copyExpression]
def leafCode : List Stmt :=
  [.ifThen improveTest improveCode [], .return (.constant .none)]
def code : List Stmt :=
  executableBody ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2

theorem code_source : code =
    .assign [.name "k"] (.builtinCall .len [.name "clique"]) ::
    .ifThen (.compare (.name "k") [(.eq, .constant (.int 5))]) leafCode [] :: code.drop 2 := rfl

theorem cells_distinct (incumbent : Incumbent heap sumCell setCell bound selected) : sumCell ≠ setCell := by
  intro same
  have impossible := incumbent.sumFound
  rw [same, incumbent.setFound] at impossible
  cases impossible

private theorem clique_ne_sum (incumbent : Incumbent heap sumCell setCell bound selected)
    (address : Nat) (clique : List Nat) (found : heap[address]? = some (.list (numbers clique)))
    (size : clique.length = 5) : sumCell ≠ address := by
  intro same
  have impossible := incumbent.sumFound
  rw [same, found] at impossible
  have lengthEq := congrArg List.length (Mutable.Object.list.inj (Option.some.inj impossible))
  simp only [numbers, primes_up_to.numbers, List.length_map, List.length_cons, List.length_nil] at lengthEq
  omega

def sumHeap (heap : Mutable.Heap) (sumCell : Nat) (sum : Int) : Mutable.Heap :=
  heap.set sumCell (.list [number sum])

def improvedHeap (heap : Mutable.Heap) (sumCell setCell : Nat) (sum : Int) (clique : List Nat) : Mutable.Heap :=
  (sumHeap heap sumCell sum ++ [Mutable.Object.list (numbers clique)]).set setCell (.list [.ref heap.length])

def leafHeap (heap : Mutable.Heap) (sumCell setCell : Nat) (sum bound : Int) (clique : List Nat) : Mutable.Heap :=
  if sum < bound then improvedHeap heap sumCell setCell sum clique else heap

private theorem read_cell (env : HeapCalls.Env) (name : String) (address : Nat) (value : Mutable.Value)
    (getCell : env name = some (.ref address)) (found : heap[address]? = some (.list [value])) :
    Evaluates program env (.subscript (.name name) (.constant (.int 0))) heap value heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCell) Evaluates.constant
  exact Mutable.getItem_list heap address 0 [value] found (by simp)

theorem improve_executes (env : HeapCalls.Env) (heap : Mutable.Heap) (sumCell setCell : Nat)
    (bound sum : Int) (selected clique : List Nat) (cliqueAddress : Nat)
    (incumbent : Incumbent heap sumCell setCell bound selected)
    (found : heap[cliqueAddress]? = some (.list (numbers clique))) (size : clique.length = 5)
    (getSum : env "curr_sum" = some (number sum))
    (getSumCell : env "__formalsw_cell_solve_best_sum" = some (.ref sumCell))
    (getSetCell : env "__formalsw_cell_solve_best_set" = some (.ref setCell))
    (getClique : env "clique" = some (.ref cliqueAddress)) :
    BlockContinues program env improveCode heap env (improvedHeap heap sumCell setCell sum clique) globals := by
  let written := sumHeap heap sumCell sum
  let copied := written ++ [.list (numbers clique)]
  have first : BlockContinues program env [improveCode[0]'(by decide)] heap env written globals := by
    apply BlockContinues.setItem (by intros; simp) (Evaluates.lookup getSum)
      (.cons (Evaluates.lookup getSumCell) (.cons Evaluates.constant .nil))
    simp [Mutable.setItem, incumbent.sumFound, written, sumHeap, Mutable.index, Except.map, number]
  have retained : written[cliqueAddress]? = some (.list (numbers clique)) := by
    simpa only [written, sumHeap, List.getElem?_set_ne (clique_ne_sum incumbent cliqueAddress clique found size)] using found
  have copiedEval : Evaluates program env copyExpression written (.ref heap.length) copied globals := by
    apply Evaluates.slice (.cons (Evaluates.lookup getClique)
      (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
    simpa [written, sumHeap] using Mutable.getSlice_list_copy written cliqueAddress (numbers clique) retained
  have setRetained : written[setCell]? = some (.list [.ref incumbent.selectedAddress]) := by
    simpa only [written, sumHeap, List.getElem?_set_ne (cells_distinct incumbent)] using incumbent.setFound
  have setCopied : copied[setCell]? = some (.list [.ref incumbent.selectedAddress]) :=
    HeapCollection.prefix_found written _ _ _ setRetained
  have second : BlockContinues program env [improveCode[1]'(by decide)] written env
      (improvedHeap heap sumCell setCell sum clique) globals := by
    apply BlockContinues.setItem (by intros; simp) copiedEval
      (.cons (Evaluates.lookup getSetCell) (.cons Evaluates.constant .nil))
    simp [Mutable.setItem, setCopied, improvedHeap, copied, written, Mutable.index, Except.map]
  simpa [improveCode] using first.append second

theorem leaf_result (env : HeapCalls.Env) (heap : Mutable.Heap) (sumCell setCell : Nat)
    (bound sum : Int) (selected clique : List Nat) (cliqueAddress : Nat)
    (incumbent : Incumbent heap sumCell setCell bound selected)
    (found : heap[cliqueAddress]? = some (.list (numbers clique))) (size : clique.length = 5)
    (getSum : env "curr_sum" = some (number sum))
    (getSumCell : env "__formalsw_cell_solve_best_sum" = some (.ref sumCell))
    (getSetCell : env "__formalsw_cell_solve_best_set" = some (.ref setCell))
    (getClique : env "clique" = some (.ref cliqueAddress)) (following : List Stmt) :
    BlockReturns program env (leafCode ++ following) heap nothing
      (leafHeap heap sumCell setCell sum bound clique) globals := by
  have tested : Evaluates program env improveTest heap (flag (decide (sum < bound))) heap globals :=
    Evaluates.compare (Evaluates.lookup getSum) (read_cell env _ _ _ getSumCell incumbent.sumFound)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag])
  by_cases improves : sum < bound
  · rw [leafHeap, if_pos improves]
    apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
      (by simpa only [improves, decide_true] using tested) rfl
    exact (improve_executes env heap sumCell setCell bound sum selected clique cliqueAddress incumbent found size
      getSum getSumCell getSetCell getClique).returns (BlockReturns.return_value Evaluates.constant)
  · rw [leafHeap, if_neg improves]
    apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
      (by simpa only [improves, decide_false] using tested) rfl
    exact BlockReturns.return_value Evaluates.constant

theorem complete_result (cliqueAddress : Nat) (candidates : Mutable.Value) (sum : Int)
    (heap : Mutable.Heap) (sumCell setCell cacheCell powerCell : Nat) (bound : Int)
    (selected clique : List Nat) (incumbent : Incumbent heap sumCell setCell bound selected)
    (found : heap[cliqueAddress]? = some (.list (numbers clique))) (size : clique.length = 5) :
    ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref cliqueAddress, candidates, number sum, .ref setCell, .ref sumCell, .ref cacheCell, .ref powerCell]
      heap nothing (leafHeap heap sumCell setCell sum bound clique) globals := by
  let params := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
  let env := functionEnv params sourceBody
    [.ref cliqueAddress, candidates, number sum, .ref setCell, .ref sumCell, .ref cacheCell, .ref powerCell] globals
  have getClique : env "clique" = some (.ref cliqueAddress) := by cases globals <;> rfl
  have lengthEval : Evaluates program env (.builtinCall .len [.name "clique"]) heap (number 5) heap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getClique) .nil)
    simp [primitive, Mutable.length, found, numbers, primes_up_to.numbers, size]
  let counted := setLocal env "k" (number 5)
  have getK : counted "k" = some (number 5) := by simp [counted, setLocal]
  apply BlockReturns.function rfl rfl
  change BlockReturns program env code heap nothing (leafHeap heap sumCell setCell sum bound clique) globals
  rw [code_source]
  apply BlockReturns.assign lengthEval
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
    (Evaluates.compare (Evaluates.lookup getK) Evaluates.constant rfl) rfl
  apply leaf_result counted heap sumCell setCell bound sum selected clique cliqueAddress incumbent found size
  · cases globals <;> rfl
  · cases globals <;> rfl
  · cases globals <;> rfl
  · simpa [counted, setLocal] using getClique

theorem insufficient_result (cliqueAddress candidateAddress : Nat) (sum : Int)
    (heap : Mutable.Heap) (sumCell setCell cacheCell powerCell : Nat) (clique candidates : List Nat)
    (cliqueFound : heap[cliqueAddress]? = some (.list (numbers clique)))
    (candidateFound : heap[candidateAddress]? = some (.list (numbers candidates)))
    (short : clique.length + candidates.length < 5) :
    ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref cliqueAddress, .ref candidateAddress, number sum, .ref setCell, .ref sumCell, .ref cacheCell, .ref powerCell]
      heap nothing heap globals := by
  let params := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
  let env := functionEnv params sourceBody
    [.ref cliqueAddress, .ref candidateAddress, number sum, .ref setCell, .ref sumCell, .ref cacheCell, .ref powerCell] globals
  let counted := setLocal env "k" (number clique.length)
  let ready := setLocal counted "need" (number (5 - (clique.length : Int)))
  have getClique : env "clique" = some (.ref cliqueAddress) := by cases globals <;> rfl
  have getK : counted "k" = some (number clique.length) := by simp [counted, setLocal]
  have getCandidates : ready "candidates" = some (.ref candidateAddress) := by cases globals <;> rfl
  have getNeed : ready "need" = some (number (5 - (clique.length : Int))) := by simp [ready, setLocal]
  have candidateLength : Evaluates program ready (.builtinCall .len [.name "candidates"])
      heap (number candidates.length) heap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getCandidates) .nil)
    simp [primitive, Mutable.length, candidateFound, numbers, primes_up_to.numbers]
  apply BlockReturns.function rfl rfl
  change BlockReturns program env code heap nothing heap globals
  rw [code_source]
  apply BlockReturns.assign (value := number clique.length) (middle := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getClique) .nil)
    simp [primitive, Mutable.length, cliqueFound, numbers, primes_up_to.numbers]
  · apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
      (Evaluates.compare (Evaluates.lookup getK) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
    apply BlockReturns.assign (value := number (5 - (clique.length : Int))) (middle := heap)
      (Evaluates.binary Evaluates.constant (Evaluates.lookup getK) rfl)
    apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
      (Evaluates.compare candidateLength (Evaluates.lookup getNeed)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
    exact BlockReturns.return_value Evaluates.constant

def outerBody : List Stmt := match (code[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private theorem fallthrough (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockReturns program env [] heap nothing heap globals := fun _ _ => .next rfl .refl

theorem first_bound_result (cliqueAddress candidateAddress : Nat) (sum : Int)
    (heap : Mutable.Heap) (sumCell setCell cacheCell powerCell : Nat) (bound : Int)
    (selected clique : List Nat) (q : Nat) (remaining : List Nat)
    (incumbent : Incumbent heap sumCell setCell bound selected)
    (cliqueFound : heap[cliqueAddress]? = some (.list (numbers clique)))
    (candidateFound : heap[candidateAddress]? = some (.list (numbers (q :: remaining))))
    (incomplete : clique.length < 5) (enough : 5 ≤ clique.length + (q :: remaining).length)
    (pruned : bound ≤ sum + (q : Int) * (5 - (clique.length : Int))) :
    ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref cliqueAddress, .ref candidateAddress, number sum, .ref setCell, .ref sumCell, .ref cacheCell, .ref powerCell]
      heap nothing heap globals := by
  simp only [List.length_cons] at enough
  let params := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
  let env := functionEnv params sourceBody
    [.ref cliqueAddress, .ref candidateAddress, number sum, .ref setCell, .ref sumCell, .ref cacheCell, .ref powerCell] globals
  let counted := setLocal env "k" (number clique.length)
  let ready := setLocal counted "need" (number (5 - (clique.length : Int)))
  let indexed := setLocal ready "idx" (number 0)
  let chosen := setLocal indexed "q" (number q)
  have getClique : env "clique" = some (.ref cliqueAddress) := by cases globals <;> rfl
  have getK : counted "k" = some (number clique.length) := by simp [counted, setLocal]
  have getCandidates : ready "candidates" = some (.ref candidateAddress) := by cases globals <;> rfl
  have getNeed : ready "need" = some (number (5 - (clique.length : Int))) := by simp [ready, setLocal]
  have candidateLength : Evaluates program ready (.builtinCall .len [.name "candidates"])
      heap (number (q :: remaining).length) heap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getCandidates) .nil)
    simp [primitive, Mutable.length, candidateFound, numbers, primes_up_to.numbers]
  have getIndex : indexed "idx" = some (number 0) := by simp [indexed, setLocal]
  have getIndexedCandidates : indexed "candidates" = some (.ref candidateAddress) := by
    simpa [indexed, setLocal] using getCandidates
  have getQ : chosen "q" = some (number q) := by simp [chosen, setLocal]
  have getChosenNeed : chosen "need" = some (number (5 - (clique.length : Int))) := by
    simpa [chosen, indexed, setLocal] using getNeed
  have getSum : chosen "curr_sum" = some (number sum) := by cases globals <;> rfl
  have getSumCell : chosen "__formalsw_cell_solve_best_sum" = some (.ref sumCell) := by cases globals <;> rfl
  have guard : Evaluates program chosen
      (.compare (.binOp (.name "curr_sum") .add (.binOp (.name "q") .mult (.name "need"))) [(.gtE, sumExpression)])
      heap (flag true) heap globals := by
    apply Evaluates.compare
      (Evaluates.binary (Evaluates.lookup getSum)
        (Evaluates.binary (Evaluates.lookup getQ) (Evaluates.lookup getChosenNeed) rfl) rfl)
      (read_cell chosen _ _ _ getSumCell incumbent.sumFound)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, pruned]
  apply BlockReturns.function rfl rfl
  change BlockReturns program env code heap nothing heap globals
  rw [code_source]
  apply BlockReturns.assign (value := number clique.length) (middle := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getClique) .nil)
    simp [primitive, Mutable.length, cliqueFound, numbers, primes_up_to.numbers]
  · apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
      (Evaluates.compare (Evaluates.lookup getK) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
    apply BlockReturns.assign (value := number (5 - (clique.length : Int))) (middle := heap)
      (Evaluates.binary Evaluates.constant (Evaluates.lookup getK) rfl)
    apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
      (Evaluates.compare candidateLength (Evaluates.lookup getNeed)
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
    apply BlockReturns.for_range (.cons candidateLength .nil) rfl
    apply BlockReturns.range_step (by simp [Calls.rangeContinues])
    apply BlockReturns.assign (value := number q) (middle := heap)
    · apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getIndexedCandidates) (Evaluates.lookup getIndex)
      exact Mutable.getItem_list heap candidateAddress 0 (numbers (q :: remaining)) candidateFound (by simp [numbers, primes_up_to.numbers])
    · apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap) guard rfl
      apply BlockReturns.break_loop rfl
      exact fallthrough chosen heap

theorem improved_preserves (heap : Mutable.Heap) (sumCell setCell : Nat) (sum : Int) (clique : List Nat)
    (address : Nat) (object : Mutable.Object) (found : heap[address]? = some object)
    (notSum : sumCell ≠ address) (notSet : setCell ≠ address) :
    (improvedHeap heap sumCell setCell sum clique)[address]? = some object := by
  unfold improvedHeap
  rw [List.getElem?_set_ne notSet]
  apply HeapCollection.prefix_found (sumHeap heap sumCell sum) _ _ _
  simpa only [sumHeap, List.getElem?_set_ne notSum] using found

def improvedIncumbent (incumbent : Incumbent heap sumCell setCell bound selected) (sum : Int) (clique : List Nat) :
    Incumbent (improvedHeap heap sumCell setCell sum clique) sumCell setCell sum clique where
  selectedAddress := heap.length
  sumFound := by
    unfold improvedHeap
    rw [List.getElem?_set_ne (Ne.symm (cells_distinct incumbent))]
    apply HeapCollection.prefix_found (sumHeap heap sumCell sum) _ _ _
    exact List.getElem?_set_self ((List.getElem?_eq_some_iff.mp incumbent.sumFound).1)
  setFound := by
    apply List.getElem?_set_self
    have old := (List.getElem?_eq_some_iff.mp incumbent.setFound).1
    simp only [sumHeap, List.length_append, List.length_set, List.length_cons, List.length_nil]
    omega
  selectedFound := by
    have old := (List.getElem?_eq_some_iff.mp incumbent.setFound).1
    unfold improvedHeap
    rw [List.getElem?_set_ne (by omega : setCell ≠ heap.length)]
    simp [sumHeap]

def leafIncumbent (incumbent : Incumbent heap sumCell setCell bound selected) (sum : Int) (clique : List Nat) :
    Incumbent (leafHeap heap sumCell setCell sum bound clique) sumCell setCell
      (CliqueSearch.improve clique sum ⟨bound, selected⟩).bound
      (CliqueSearch.improve clique sum ⟨bound, selected⟩).chosen := by
  by_cases better : sum < bound
  · simpa only [leafHeap, CliqueSearch.improve, if_pos better] using improvedIncumbent incumbent sum clique
  · simpa only [leafHeap, CliqueSearch.improve, if_neg better] using incumbent

/-- The source's strict update has exactly the value represented by the zero-
remaining-picks case of the recursive search model. -/
theorem complete_model (relation : Nat → Nat → Bool) (candidates clique : List Nat) (sum bound : Int)
    (selected : List Nat) :
    CliqueSearch.search relation 0 clique candidates sum ⟨bound, selected⟩ =
      (if sum < bound then ⟨sum, clique⟩ else ⟨bound, selected⟩) := rfl

theorem insufficient_model (relation : Nat → Nat → Bool) (clique candidates : List Nat)
    (sum : Int) (best : CliqueSearch.Best) (short : clique.length + candidates.length < 5) :
    CliqueSearch.search relation (5 - clique.length) clique candidates sum best = best :=
  CliqueSearch.search_insufficient relation _ _ _ _ _ (by omega) (by omega)

theorem first_bound_model (relation : Nat → Nat → Bool) (clique : List Nat) (q : Nat) (remaining : List Nat)
    (sum : Int) (best : CliqueSearch.Best) (incomplete : clique.length < 5)
    (enough : 5 ≤ clique.length + (q :: remaining).length)
    (cut : best.bound ≤ sum + (q : Int) * (5 - (clique.length : Int))) :
    CliqueSearch.search relation (5 - clique.length) clique (q :: remaining) sum best = best := by
  apply CliqueSearch.search_head_pruned relation _ _ _ _ _ _ (by omega) (by omega)
  have castNeed : ((5 - clique.length : Nat) : Int) = 5 - (clique.length : Int) := by omega
  simpa only [castNeed] using cut

variable {left right address target : Nat} {values tupleValues : List Mutable.Value}
  {entries : List (Constant × Mutable.Value)}

private theorem list_ne_dict (listFound : heap[left]? = some (.list values))
    (dictFound : heap[right]? = some (.dict entries)) : left ≠ right := by
  intro same
  rw [same, dictFound] at listFound
  cases listFound

private theorem list_ne_tuple (listFound : heap[left]? = some (.list values))
    (tupleFound : heap[right]? = some (.tuple tupleValues)) : left ≠ right := by
  intro same
  rw [same, tupleFound] at listFound
  cases listFound

private theorem sum_ne_ref_cell (incumbent : Incumbent heap sumCell setCell bound selected)
    (found : heap[address]? = some (.list [.ref target])) : sumCell ≠ address := by
  intro same
  have impossible := incumbent.sumFound
  rw [same, found] at impossible
  cases impossible

private theorem set_ne_ref_cell (incumbent : Incumbent heap sumCell setCell bound selected)
    (found : heap[address]? = some (.list [.ref target]))
    (different : target ≠ incumbent.selectedAddress) : setCell ≠ address := by
  intro same
  have observed : heap[setCell]? = some (.list [.ref target]) := by simpa only [same] using found
  have refs := observed.symm.trans incumbent.setFound
  exact different (by simpa using refs)

def cache_after_improve (incumbent : Incumbent heap sumCell setCell bound selected)
    (cacheCell powerCell : Nat) (entries : List IntPairCache.Entry)
    (store : pair_ok.Store heap entries cacheCell powerCell) (sum : Int) (clique : List Nat) :
    pair_ok.Store (improvedHeap heap sumCell setCell sum clique) entries cacheCell powerCell where
  cacheAddress := store.cacheAddress
  powerAddress := store.powerAddress
  powers := store.powers
  cacheCellFound := improved_preserves heap sumCell setCell sum clique _ _ store.cacheCellFound
    (sum_ne_ref_cell incumbent store.cacheCellFound)
    (set_ne_ref_cell incumbent store.cacheCellFound (store.cache_ne_list incumbent.selectedFound))
  powerCellFound := improved_preserves heap sumCell setCell sum clique _ _ store.powerCellFound
    (sum_ne_ref_cell incumbent store.powerCellFound)
    (set_ne_ref_cell incumbent store.powerCellFound (Ne.symm (list_ne_dict incumbent.selectedFound store.powerFound)))
  cacheFound := improved_preserves heap sumCell setCell sum clique _ _ store.cacheFound
    (Ne.symm (store.cache_ne_list incumbent.sumFound)) (Ne.symm (store.cache_ne_list incumbent.setFound))
  powerFound := improved_preserves heap sumCell setCell sum clique _ _ store.powerFound
    (list_ne_dict incumbent.sumFound store.powerFound) (list_ne_dict incumbent.setFound store.powerFound)
  separate := store.separate

def globals_after_improve (incumbent : Incumbent heap sumCell setCell bound selected)
    (context : is_prime.SourceGlobals heap globals) (sum : Int) (clique : List Nat) :
    is_prime.SourceGlobals (improvedHeap heap sumCell setCell sum clique) globals where
  smallAddress := context.smallAddress
  basesAddress := context.basesAddress
  smallBinding := context.smallBinding
  basesBinding := context.basesBinding
  smallTuple := improved_preserves heap sumCell setCell sum clique _ _ context.smallTuple
    (list_ne_tuple incumbent.sumFound context.smallTuple) (list_ne_tuple incumbent.setFound context.smallTuple)
  basesTuple := improved_preserves heap sumCell setCell sum clique _ _ context.basesTuple
    (list_ne_tuple incumbent.sumFound context.basesTuple) (list_ne_tuple incumbent.setFound context.basesTuple)

end solvers.«60».dfs
