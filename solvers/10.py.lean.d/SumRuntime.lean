import solvers.«10.py.lean.d».Source
import lang.py.FormalSW.HeapRuntimeCostBufferLoops
import lang.py.FormalSW.HeapRuntimeCostRanges

namespace solvers.«10».sumRuntime
open FormalSW.PyAst HeapCalls.Runtime
private abbrev program := Solvers.P10Ast.module

def solveBody : List Stmt :=
  match findFunctionIn program.body "solve" with
  | some (_, body) => executableBody body
  | none => []

def sumBody : List Stmt :=
  match (solveBody[8]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

theorem loop (heap : Mutable.Heap) (address : Nat) (bytes : List UInt8)
    (found : heap[address]? = some (.bytearray bytes)) (position : Nat) (bounded : position ≤ bytes.length)
    (total : Int) (env : HeapCalls.Env) (getTotal : env "total" = some (.scalar (.int total))) :
    ∃ finalEnv, BlockContinuesIn .word program env
      [.enumerateStepLoop "i" "is_prime" (.ref address) position position sumBody []] heap
      (27 * (bytes.length - position) + 7) finalEnv heap ∧
      finalEnv "total" = some (.scalar (.int (total + (oddSum position (bytes.drop position) : Int)))) := by
  have sequence : Mutable.sequence (.ref address) heap =
      some (.ok (bytes.map (fun byte => Mutable.Value.scalar (.int byte.toNat))), heap) := by
    simp [Mutable.sequence, found]
  by_cases active : position < bytes.length
  · let byte := bytes[position]
    let boundEnv := HeapCalls.setLocal (HeapCalls.setLocal env "i" (.scalar (.int position)))
      "is_prime" (.scalar (.int byte.toNat))
    have saved : boundEnv "total" = some (.scalar (.int total)) := by
      simpa [boundEnv, HeapCalls.setLocal] using getTotal
    have iteration : ∃ middleEnv, BlockContinuesIn .word program boundEnv sumBody heap 20 middleEnv heap ∧
        middleEnv "total" = some (.scalar (.int (total + contribution position byte))) := by
      by_cases zero : byte.toNat = 0
      · refine ⟨boundEnv, ?_, ?_⟩
        · apply BlockContinuesIn.mono (small := 3) _ (by decide)
          apply BlockContinuesIn.branch (cc := 1) (cb := 0) (flag := false)
            (EvaluatesIn.lookup (env := boundEnv) (name := "is_prime") rfl)
          · simp [Mutable.truth, scalarTruth, zero]
          · exact BlockContinuesIn.nil
        · simpa [contribution, zero] using saved
      · refine ⟨HeapCalls.setLocal boundEnv "total" (.scalar (.int (total + contribution position byte))), ?_, rfl⟩
        apply BlockContinuesIn.branch (cc := 1) (cb := 17) (flag := true)
          (EvaluatesIn.lookup (name := "is_prime") rfl)
        · simp [Mutable.truth, scalarTruth, zero]
        · apply BlockContinuesIn.augAssign (cost := 11) (work := 1) (right := .int (oddValue position)) saved
          · apply EvaluatesIn.binary (ca := 6) (cb := 1) (work := 1)
              (av := .int (2 * position : Nat)) (bv := .int 3)
            · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1) (av := .int 2) (bv := .int position)
              · exact EvaluatesIn.constant
              · exact EvaluatesIn.lookup (by simp [boundEnv, HeapCalls.setLocal])
              · simp [Mutable.binary, scalarBinary]
              · rfl
            · exact EvaluatesIn.constant
            · simp [Mutable.binary, scalarBinary, oddValue]
            · rfl
          · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, contribution, zero]
          · rfl
    obtain ⟨middleEnv, iterationRun, savedNext⟩ := iteration
    obtain ⟨finalEnv, rest, answer⟩ := loop heap address bytes found (position + 1) (by omega)
      (total + contribution position byte) middleEnv savedNext
    refine ⟨finalEnv, ?_, ?_⟩
    · have entered := BlockContinuesIn.buffer_enumerate_step (env := env)
        (position := position) (counter := (position : Int)) (index := "i") (name := "is_prime")
        (value := .scalar (.int byte.toNat)) (body := sumBody) (otherwise := []) found sequence
        (by simp [List.getElem?_map, List.getElem?_eq_getElem active, byte])
        (by simpa only [Nat.cast_add, Nat.cast_one] using iterationRun.append rest)
      exact entered.mono (by omega)
    · have tail : bytes.drop position = byte :: bytes.drop (position + 1) := List.drop_eq_getElem_cons active
      simpa only [tail, oddSum, Nat.cast_add, Int.add_assoc] using answer
  · have equal : position = bytes.length := by omega
    refine ⟨env, ?_, ?_⟩
    · simpa [equal] using BlockContinuesIn.buffer_enumerate_done (module := program) (env := env)
        (index := "i") (name := "is_prime") (counter := (position : Int)) (body := sumBody)
        (position := position) found sequence (by simp [equal]) BlockContinuesIn.nil
    · simpa [equal, oddSum] using getTotal
termination_by bytes.length - position
decreasing_by omega

theorem finish (bytes : List UInt8) (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (getSieve : env "sieve" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    BlockReturnsIn .word program env (solveBody.drop 7) heap (27 * bytes.length + 19)
      (.scalar (.int (2 + (oddSum 0 bytes : Int)))) heap := by
  let initialized := HeapCalls.setLocal env "total" (.scalar (.int 2))
  obtain ⟨finalEnv, loopRun, answer⟩ := loop heap address bytes found 0 (by omega) 2 initialized rfl
  have ranged : BlockContinuesIn .word program initialized [solveBody[8]'(by decide)]
      heap (27 * bytes.length + 14) finalEnv heap := by
    have run := BlockContinuesIn.for_buffer_enumerate (index := "i") (name := "is_prime")
      (items := bytes.map fun byte => .scalar (.int byte.toNat))
      (EvaluatesIn.lookup (name := "sieve") (by simpa [initialized, HeapCalls.setLocal] using getSieve)) found
      (by simp [Mutable.sequence, found]) loopRun
    exact run.mono (by omega)
  have returned : BlockReturnsIn .word program finalEnv [solveBody[9]'(by decide)] heap 2
      (.scalar (.int (2 + (oddSum 0 bytes : Int)))) heap :=
    BlockReturnsIn.return_value (EvaluatesIn.lookup (by simpa using answer))
  have result := (BlockContinuesIn.assign (name := "total") (EvaluatesIn.constant
    (arithmetic := .word) (module := program) (env := env) (heap := heap) (value := .int 2))).returns
      (ranged.returns returned)
  exact result.mono (by omega)

end solvers.«10».sumRuntime
