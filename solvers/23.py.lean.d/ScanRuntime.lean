import solvers.«23.py.lean.d».MarkingRuntime

/-! The final source scan pays for a byte read and an optional integer addition
at each index. No correctness assumption about the marking buffer is needed. -/
namespace solvers.«23».runtimeCost.Scan
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
abbrev program := Solvers.P23Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

def body : List Stmt :=
  [.ifThen (.unaryOp .not (.subscript (.name "can") (.name "n")))
    [.augAssign (.name "total") .add (.name "n")] []]

def statements : List Stmt :=
  [.assign [.name "total"] (.constant (.int 0)),
   .forIn (.name "n") (.call (.name "range")
     [.constant (.int 1), .binOp (.name "lim") .add (.constant (.int 1))]) body [],
   .return (.name "total")]

theorem tested (bytes : List UInt8) (position address : Nat) (inside : position < bytes.length)
    (heap : Mutable.Heap) (found : heap[address]? = some (.bytearray bytes))
    (env : HeapCalls.Env) (getCan : env "can" = some (.ref address))
    (getN : env "n" = some (number position)) :
    EvaluatesIn .word program env (.unaryOp .not (.subscript (.name "can") (.name "n")))
      heap 12 (.scalar (.bool (decide (bytes[position].toNat = 0)))) heap := by
  have read : EvaluatesIn .word program env (.subscript (.name "can") (.name "n")) heap 10
      (number bytes[position].toNat) heap :=
    EvaluatesIn.buffer_subscript (by intro a b c equal; cases equal)
      (EvaluatesIn.lookup getCan) (EvaluatesIn.lookup getN) found
      (Mutable.getItem_buffer heap address position bytes found inside)
  intro anchors stack
  have first : Reaches .word program anchors
      ⟨.expr (.unaryOp .not (.subscript (.name "can") (.name "n"))) env, stack, heap⟩ 1
      ⟨.expr (.subscript (.name "can") (.name "n")) env, .unary .not :: stack, heap⟩ :=
    advance rfl rfl rfl
  have last : Reaches .word program anchors
      ⟨.value (number bytes[position].toNat), .unary .not :: stack, heap⟩ 1
      ⟨.value (.scalar (.bool (decide (bytes[position].toNat = 0)))), stack, heap⟩ := by
    apply advance
    · by_cases zero : bytes[position].toNat = 0 <;>
        simp [step, result, run, HeapCalls.unary, Mutable.truth, scalarTruth, zero]
    · rfl
    · rfl
  exact (first.trans (read anchors _)).trans last

theorem loop (bytes : List UInt8) (limit count start total address : Nat)
    (size : bytes.length = limit + 1) (stop : start + count = limit + 1)
    (heap : Mutable.Heap) (found : heap[address]? = some (.bytearray bytes))
    (env : HeapCalls.Env) (getCan : env "can" = some (.ref address))
    (getTotal : env "total" = some (number total)) :
    ∃ finalTotal finalEnv, finalEnv "total" = some (number finalTotal) ∧
      BlockContinuesIn .word program env [.rangeStepLoop "n" start (limit + 1) 1 body []]
        heap (25 * count + 4) finalEnv heap := by
  induction count generalizing start total env with
  | zero =>
      refine ⟨total, env, getTotal, ?_⟩
      exact BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
  | succ count ih =>
      let boundEnv := setLocal env "n" (number start)
      have savedCan : boundEnv "can" = some (.ref address) := by
        simpa [boundEnv, setLocal] using getCan
      have savedTotal : boundEnv "total" = some (number total) := by
        simpa [boundEnv, setLocal] using getTotal
      have inside : start < bytes.length := by omega
      have test := tested bytes start address inside heap found boundEnv savedCan rfl
      have iteration : ∃ nextTotal nextEnv,
          nextEnv "total" = some (number nextTotal) ∧
          nextEnv "can" = some (.ref address) ∧
          BlockContinuesIn .word program boundEnv body heap 21 nextEnv heap := by
        by_cases zero : bytes[start].toNat = 0
        · refine ⟨total + start, setLocal boundEnv "total" (number (total + start)), rfl, ?_, ?_⟩
          · simpa [setLocal] using savedCan
          · apply BlockContinuesIn.branch (cc := 12) (cb := 7) (flag := true)
              (by simpa [zero] using test) rfl
            apply BlockContinuesIn.augAssign (work := 1) savedTotal (EvaluatesIn.lookup (name := "n") rfl)
            · simp [inplace, Mutable.binary, scalarBinary, number]
            · rfl
        · refine ⟨total, boundEnv, savedTotal, savedCan, ?_⟩
          have skipped := BlockContinuesIn.branch
            (yes := [.augAssign (.name "total") .add (.name "n")]) (no := [])
            (flag := false) (by simpa [zero] using test) rfl BlockContinuesIn.nil
          exact skipped.mono (by omega)
      obtain ⟨nextTotal, nextEnv, nextTotalLookup, nextCan, iterationRun⟩ := iteration
      obtain ⟨finalTotal, finalEnv, finalLookup, rest⟩ :=
        ih (start + 1) nextTotal (by omega) nextEnv nextCan nextTotalLookup
      refine ⟨finalTotal, finalEnv, finalLookup, ?_⟩
      have run := BlockContinuesIn.range_step (env := env) (name := "n")
        (next := (start : Int)) (stop := (limit + 1 : Nat)) (stride := 1)
        (body := body) (otherwise := []) (by simp [Calls.rangeContinues]; omega)
        (by simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          boundEnv, number] using iterationRun.append rest)
      exact run.mono (by omega)

theorem source_run (bytes : List UInt8) (limit address : Nat) (size : bytes.length = limit + 1)
    (heap : Mutable.Heap) (found : heap[address]? = some (.bytearray bytes))
    (env : HeapCalls.Env) (getCan : env "can" = some (.ref address))
    (getLimit : env "lim" = some (number limit)) :
    ∃ total, BlockReturnsIn .word program env statements heap (25 * limit + 23) (number total) heap := by
  let initialized := setLocal env "total" (number 0)
  have initial_run : BlockContinuesIn .word program env
      [.assign [.name "total"] (.constant (.int 0))] heap 3 initialized heap :=
    BlockContinuesIn.assign EvaluatesIn.constant
  obtain ⟨total, finalEnv, totalLookup, run⟩ := loop bytes limit limit 1 0 address size (by omega)
    heap found initialized (by simpa [initialized, setLocal] using getCan) rfl
  have upper : EvaluatesIn .word program initialized
      (.binOp (.name "lim") .add (.constant (.int 1))) heap 6 (number (limit + 1)) heap := by
    apply EvaluatesIn.binary (work := 1)
      (EvaluatesIn.lookup (by simpa [initialized, setLocal] using getLimit)) EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, number]
    · rfl
  have ranged := BlockContinuesIn.for_range_pair (EvaluatesIn.constant (value := .int 1)) upper run
  have returned : BlockReturnsIn .word program finalEnv [.return (.name "total")]
      heap 2 (number total) heap := BlockReturnsIn.return_value (EvaluatesIn.lookup totalLookup)
  refine ⟨total, ?_⟩
  exact (initial_run.returns (ranged.returns returned)).mono (by omega)

end solvers.«23».runtimeCost.Scan
