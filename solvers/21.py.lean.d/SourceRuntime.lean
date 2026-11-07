import solvers.«21.py.lean.d».ScanRuntime
import solvers.«21.py.lean.d».SieveResult
import lang.py.FormalSW.HeapRuntimeCostSetup

namespace solvers.«21».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open ProjectEulerStatements.P21

abbrev program := Solvers.P21Ast.module
def number (value : Int) : Mutable.Value := .scalar (.int value)
def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])

theorem function_body_eq : executableBody definition.2 =
    [.assign [.name "divsum"] (.call (.name "proper_divisor_sums_sieve")
      [.binOp (.name "limit") .sub (.constant (.int 1))]),
     .assign [.name "total"] (.constant (.int 0)),
     .forIn (.name "a") (.call (.name "range") [.constant (.int 2), .name "limit"]) Scan.body [],
     .return (.name "total")] := rfl

/-- Compose allocation, sieve, cached factorization and every final scan pass.
Independent functional results identify the table and factorization answers. -/
theorem source_bound (limit : Nat) (large : 2 ≤ limit)
    (sieveCorrect : ReturnsValue program "proper_divisor_sums_sieve" [number (limit - 1 : Nat)] []
      (.ref 1) [.list [number 0], .list (solvers.«21».Sieve.table limit d)])
    (factorCorrect : ∀ (x : Nat) (heap : Mutable.Heap),
      ReturnsValue program "proper_divisor_sum_factorization" [number x] heap (number (d x)) heap) :
    TimeBound .word program "solve" [number limit] [] (1000 * (limit + 1)^2) := by
  let entry := functionEnv definition.1 definition.2 [number limit] []
  let allocated : Mutable.Heap := [.list [number 0], .list (solvers.«21».Sieve.table limit d)]
  let withTable := setLocal entry "divsum" (.ref 1)
  let initial := setLocal withTable "total" (number 0)
  have sieve : ReturnsIn .word program "proper_divisor_sums_sieve" [number (limit - 1 : Nat)] []
      (64 * (limit * (limit.log2 + 2))) (.ref 1) allocated := by
    simpa only [Nat.sub_add_cancel (by omega : 1 ≤ limit)] using
      Sieve.function_known (limit - 1) [] (by
        simpa only [Nat.sub_add_cancel (by omega : 1 ≤ limit)] using sieveCorrect)
  have argument : EvaluatesIn .word program entry
      (.binOp (.name "limit") .sub (.constant (.int 1))) [] 6
      (number (limit - 1 : Nat)) [] :=
    EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup (name := "limit") rfl) EvaluatesIn.constant
      (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub (by omega : 1 ≤ limit)]) rfl
  have allocate := BlockContinuesIn.assign (name := "divsum")
    (EvaluatesIn.call_single argument sieve)
  have initTotal : BlockContinuesIn .word program withTable
      [.assign [.name "total"] (.constant (.int 0))] allocated 3 initial allocated :=
    BlockContinuesIn.assign EvaluatesIn.constant
  have locals : Scan.Locals initial limit 1 0 := ⟨rfl, rfl, rfl⟩
  have found : allocated[1]? = some (.list (solvers.«21».Sieve.table limit d)) := rfl
  obtain ⟨finalTotal, finalEnv, finalLocals, scan⟩ :=
    Scan.loop_cost (limit - 2) 2 limit 1 0 (by omega) allocated initial locals found factorCorrect
  have forLoop := BlockContinuesIn.for_range_pair
    (EvaluatesIn.constant (value := .int 2)) (EvaluatesIn.lookup locals.bound) scan
  have returned : BlockReturnsIn .word program finalEnv [.return (.name "total")]
      allocated 2 (number finalTotal) allocated :=
    BlockReturnsIn.return_value (EvaluatesIn.lookup finalLocals.accumulated)
  have body := allocate.returns (initTotal.returns (forLoop.returns returned))
  have called := BlockReturnsIn.function (module := program) (name := "solve")
    (args := definition.1) (body := definition.2) (values := [number limit]) rfl rfl
    (by simpa only [function_body_eq, List.cons_append, List.nil_append] using body)
  apply called.timeBound_append.mono
  change 3 + ((6 + 64 * (limit * (limit.log2 + 2)) + 2 + 2) +
    (3 + ((1 + 1 + ((700 * (limit + 1) + 4) * (limit - 2) + 4) + 7) + 2))) + 1 ≤ _
  have logarithm := Nat.log2_le_self limit
  have loops := Nat.mul_le_mul_left (700 * (limit + 1) + 4) (Nat.sub_le limit 2)
  nlinarith

end solvers.«21».runtimeCost
