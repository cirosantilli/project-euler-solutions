import solvers.«21.py.lean.d».FactorFunctionRuntime
import solvers.«21.py.lean.d».SieveTable
import ProjectEulerStatements.Lib.DivisorSumBounds
import lang.py.FormalSW.HeapRuntimeCostComparisons

/-! Runtime of the source local divisor-sum helper. Functional correctness of
the factorization fallback is supplied separately for the same input and heap. -/
namespace solvers.«21».runtimeCost.Cached
open FormalSW.PyAst HeapCalls.Runtime
open ProjectEulerStatements.P21

abbrev program := Solvers.P21Ast.module
def number (value : Int) : Mutable.Value := .scalar (.int value)

def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "__formalsw_local_solve_d").getD ([], [])

def condition : Expr :=
  match ((executableBody definition.2)[0]? : Option Stmt) with
  | some (.ifThen test _ _) => test
  | _ => .constant .none

theorem condition_eq : condition = .compare (.constant (.int 0))
    [(.ltE, .name "x"), (.lt, .builtinCall .len [.name "divsum"])] := rfl

theorem function_body_eq : executableBody definition.2 =
    [.ifThen condition [.return (.subscript (.name "divsum") (.name "x"))] [],
      .return (.call (.name "proper_divisor_sum_factorization") [.name "x"])] := rfl

theorem test_cost (x size address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (input : env "x" = some (number x))
    (array : env "divsum" = some (.ref address))
    (found : heap[address]? = some (.list (solvers.«21».Sieve.table size d))) :
    EvaluatesIn .word program env condition heap 17
      (.scalar (.bool (decide (x < size)))) heap := by
  have length : EvaluatesIn .word program env (.builtinCall .len [.name "divsum"])
      heap 5 (number size) heap := by
    apply EvaluatesIn.builtin_single (builtin := .len) (work := 1)
      (by intro equal; cases equal) (by intro equal; cases equal) (EvaluatesIn.lookup array)
    · simp [HeapCalls.primitive, Mutable.length, found, number]
    · simp [operation, sequenceLength, found]
  rw [condition_eq]
  simpa [number] using
    EvaluatesIn.nonnegative_below (EvaluatesIn.lookup input) length

/-- A hit reads one table reference. A miss executes the original factorization
helper and uses its independent functional result to identify the answer. -/
theorem function_run (x bound size address : Nat) (square : x ≤ bound * bound)
    (heap : Mutable.Heap)
    (found : heap[address]? = some (.list (solvers.«21».Sieve.table size d)))
    (correct : HeapCalls.ReturnsValue program "proper_divisor_sum_factorization" [number x]
      heap (number (d x)) heap) :
    ReturnsIn .word program "__formalsw_local_solve_d" [number x, .ref address] heap
      (300 * (bound + 1)) (number (d x)) heap := by
  let entry := HeapCalls.functionEnv definition.1 definition.2 [number x, .ref address] []
  have input : entry "x" = some (number x) := rfl
  have array : entry "divsum" = some (.ref address) := rfl
  have tested := test_cost x size address heap entry input array found
  by_cases inside : x < size
  · have read : EvaluatesIn .word program entry (.subscript (.name "divsum") (.name "x"))
        heap 10 (number (d x)) heap := by
      apply EvaluatesIn.list_subscript (ca := 1) (cb := 1)
        (by intro a b c equal; cases equal) (EvaluatesIn.lookup array) (EvaluatesIn.lookup input) found
      have item := solvers.«21».Sieve.table_lookup size d x inside
      simpa only [(List.getElem?_eq_some_iff.mp item).2] using
        Mutable.getItem_list heap address x _ found (by simpa using inside)
    apply ReturnsIn.mono (small := 35)
    · apply BlockReturnsIn.function (args := definition.1) (body := definition.2)
        (values := [number x, .ref address]) (completed := [number x, .ref address])
        (budget := 30) rfl rfl
      rw [function_body_eq]
      exact BlockReturnsIn.branch (cc := 17) (cb := 11) (flag := true)
        (by simpa [inside] using tested) rfl (BlockReturnsIn.return_value read)
    · omega
  · obtain ⟨value, factorRun⟩ := Factors.function_bound x bound square heap
    have factorKnown : ReturnsIn .word program "proper_divisor_sum_factorization" [number x] heap
        (250 * (bound + 1)) (number (d x)) heap :=
      factorRun.with_result correct
    have called := EvaluatesIn.call_single (EvaluatesIn.lookup input) factorKnown
    apply ReturnsIn.mono
      (small := 5 + (17 + ((1 + (250 * (bound + 1)) + 2) + 1) + 2))
    · apply BlockReturnsIn.function (args := definition.1) (body := definition.2)
        (completed := [number x, .ref address]) rfl rfl
      rw [function_body_eq]
      exact BlockReturnsIn.branch (cc := 17) (flag := false)
        (by simpa [inside] using tested) rfl (BlockReturnsIn.return_value called)
    · omega

end solvers.«21».runtimeCost.Cached
