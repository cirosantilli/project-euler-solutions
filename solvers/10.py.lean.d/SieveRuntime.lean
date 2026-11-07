import solvers.«10.py.lean.d».MarkingRuntime
import Mathlib.Algebra.BigOperators.Intervals

namespace solvers.«10».sieveRuntime
open FormalSW.PyAst HeapCalls.Runtime
open scoped BigOperators
private abbrev program := Solvers.P10Ast.module

def iterationBlock : List Stmt :=
  match findFunctionIn program.body "solve" with
  | some (_, body) => match ((executableBody body)[6]? : Option Stmt) with
    | some (.forIn _ _ loop _) => loop
    | _ => []
  | none => []

def candidateWork (size index : Nat) : Nat := 6 * (size / (index + 1) + 1) + 316
def loopWork (size first stop : Nat) : Nat := (∑ index ∈ Finset.Ico first stop, candidateWork size index) + 4

theorem iteration (index : Nat) (bytes : List UInt8) (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (valid : index < bytes.length) (inside : squareIndex index < bytes.length)
    (getIndex : env "i" = some (.scalar (.int index)))
    (getSize : env "size" = some (.scalar (.int bytes.length)))
    (getSieve : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv finalHeap finalBytes,
      BlockContinuesIn .word program env iterationBlock heap
        (6 * (bytes.length / (index + 1) + 1) + 312) finalEnv finalHeap ∧
      finalHeap[address]? = some (.bytearray finalBytes) ∧ finalBytes.length = bytes.length ∧
      finalEnv "size" = env "size" ∧ finalEnv "sieve" = env "sieve" := by
  have read : EvaluatesIn .word program env (.subscript (.name "sieve") (.name "i")) heap
      10 (.scalar (.int bytes[index].toNat)) heap :=
    EvaluatesIn.buffer_subscript (by intros a b c impossible; cases impossible)
      (EvaluatesIn.lookup getSieve) (EvaluatesIn.lookup getIndex) found
      (Mutable.getItem_buffer heap address index bytes found valid)
  by_cases zero : bytes[index].toNat = 0
  · refine ⟨env, heap, bytes, ?_, found, rfl, rfl, rfl⟩
    exact (BlockContinuesIn.branch (yes := markingWork.sourceBlock) (no := []) read
      (flag := false) (by simp [Mutable.truth, scalarTruth, zero]) BlockContinuesIn.nil).mono (by omega)
  · obtain ⟨finalEnv, marked, sizeStable, sieveStable⟩ :=
      markingRuntime.source_mark env heap address index bytes inside getIndex getSize getSieve found
    refine ⟨finalEnv, heap.set address (.bytearray (mark bytes index)), mark bytes index, ?_,
      List.getElem?_set_self (List.getElem?_eq_some_iff.mp found).1, mark_length bytes index, sizeStable, sieveStable⟩
    have branch := BlockContinuesIn.branch (yes := markingWork.sourceBlock) (no := []) read
      (flag := true) (by simp [Mutable.truth, scalarTruth, zero]) marked
    apply branch.mono
    have writes := markingWork.writes_le bytes.length index
    have divisor := Nat.div_le_div_left (a := bytes.length)
      (show index + 1 ≤ oddValue index by unfold oddValue; omega) (by omega : 0 < index + 1)
    omega

theorem loop (limit next : Nat) (bytes : List UInt8) (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (bounded : next ≤ (Nat.sqrt (limit - 1) - 1) / 2)
    (length : bytes.length = sieveSize limit)
    (getSize : env "size" = some (.scalar (.int bytes.length)))
    (getSieve : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv finalHeap finalBytes,
      BlockContinuesIn .word program env
        [.rangeStepLoop "i" next ((Nat.sqrt (limit - 1) - 1) / 2 : Nat) 1 iterationBlock []]
        heap (loopWork bytes.length next ((Nat.sqrt (limit - 1) - 1) / 2)) finalEnv finalHeap ∧
      finalHeap[address]? = some (.bytearray finalBytes) ∧ finalBytes.length = bytes.length ∧
      finalEnv "size" = env "size" ∧ finalEnv "sieve" = env "sieve" := by
  by_cases active : next < (Nat.sqrt (limit - 1) - 1) / 2
  · have rootBound := (candidate_index (Nat.sqrt (limit - 1)) next).mp active
    have square := Nat.le_sqrt'.mp rootBound
    have rootBelow := Nat.sqrt_le_self (limit - 1)
    have lower : 3 ≤ oddValue next := by unfold oddValue; omega
    have valid : next < bytes.length := by
      rw [length, represented_iff]
      omega
    have inside : squareIndex next < bytes.length := by
      rw [length, represented_iff, squareIndex_value]
      omega
    let indexed := HeapCalls.setLocal env "i" (.scalar (.int next))
    have indexedSize : indexed "size" = env "size" := by simp [indexed, HeapCalls.setLocal]
    have indexedSieve : indexed "sieve" = env "sieve" := by simp [indexed, HeapCalls.setLocal]
    obtain ⟨middleEnv, middleHeap, middleBytes, iterationRun, middleFound, middleLength, sizeStable, sieveStable⟩ :=
      iteration next bytes indexed heap address valid inside rfl
        (indexedSize.trans getSize) (indexedSieve.trans getSieve) found
    obtain ⟨finalEnv, finalHeap, finalBytes, rest, finalFound, finalLength, finalSize, finalSieve⟩ :=
      loop limit (next + 1) middleBytes middleEnv middleHeap address (by omega) (middleLength.trans length)
        (by rw [middleLength]; exact sizeStable.trans (indexedSize.trans getSize))
        (sieveStable.trans (indexedSieve.trans getSieve)) middleFound
    refine ⟨finalEnv, finalHeap, finalBytes, ?_, finalFound, finalLength.trans middleLength,
      finalSize.trans (sizeStable.trans indexedSize), finalSieve.trans (sieveStable.trans indexedSieve)⟩
    have combined := iterationRun.append rest
    rw [middleLength] at combined
    have entered := BlockContinuesIn.range_step (name := "i") (env := env) (next := (next : Int))
      (stop := (((Nat.sqrt (limit - 1) - 1) / 2 : Nat) : Int)) (stride := 1)
      (by simp [Calls.rangeContinues]; omega)
      (by simpa only [Nat.cast_add, Nat.cast_one] using combined)
    apply entered.mono
    unfold loopWork
    rw [Finset.sum_eq_sum_Ico_succ_bot active]
    unfold candidateWork
    omega
  · have equal : next = (Nat.sqrt (limit - 1) - 1) / 2 := by omega
    refine ⟨env, heap, bytes, ?_, found, rfl, rfl, rfl⟩
    simpa [loopWork, equal] using
      (BlockContinuesIn.range_done (module := program) (env := env) (heap := heap)
        (name := "i") (next := (next : Int)) (stop := (((Nat.sqrt (limit - 1) - 1) / 2 : Nat) : Int))
        (stride := 1) (body := iterationBlock) (otherwise := [])
        (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil)
termination_by (Nat.sqrt (limit - 1) - 1) / 2 - next
decreasing_by omega

theorem loopWork_bound (size stop : Nat) (bounded : stop ≤ size) :
    loopWork size 0 stop ≤ 6 * size * (size.log2 + 2) + 316 * size + 4 := by
  have harmonic := ProjectEulerStatements.ReciprocalSum.with_overhead size stop bounded
  simp only [loopWork, candidateWork, Nat.Ico_zero_eq_range, Finset.sum_add_distrib, ← Finset.mul_sum,
    Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul] at harmonic
  nlinarith

end solvers.«10».sieveRuntime
