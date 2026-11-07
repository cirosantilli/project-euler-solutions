import solvers.«7.py.lean.d».MarkingWork
import lang.py.FormalSW.HeapRuntimeCostBuffers
import lang.py.FormalSW.HeapRuntimeCostRanges

/-! Complete runtime costs for the source marking branch. These include scalar
expressions, RHS allocation, argument evaluation and the slice write. -/
namespace solvers.«7».markingRuntime
open FormalSW.PyAst HeapCalls.Runtime
open markingWork ProjectEulerSolutions.P7
private abbrev program := Solvers.P7Ast.module

def Stable (before after : HeapCalls.Env) : Prop :=
  ∀ name ∈ ["n", "limit", "size", "is_prime", "math"], after name = before name

theorem Stable.refl (env : HeapCalls.Env) : Stable env env := fun _ _ => rfl

theorem Stable.trans (first : Stable a b) (second : Stable b c) : Stable a c :=
  fun name member => (second name member).trans (first name member)

theorem Stable.index (env : HeapCalls.Env) (index : Nat) :
    Stable env (HeapCalls.setLocal env "i" (.scalar (.int index))) := by
  intro name member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;> rfl

/-- Empty extended slices have a conservative full-buffer deletion tariff. -/
def emptyWork (size index : Nat) : Nat := if writes size index = 0 then 5*size else 0

theorem emptyWork_of_inside (size index : Nat) (inside : oddAt index * oddAt index / 2 < size) :
    emptyWork size index = 0 := by
  have nonnegative : 0 ≤ ((size : Int) - 1 - (oddAt index * oddAt index / 2 : Nat)) / oddAt index :=
    Int.ediv_nonneg (by omega) (by positivity)
  have count := source_count size index
  have nonzero : writes size index ≠ 0 := by omega
  simp [emptyWork, nonzero]

/-- Only the last candidate of an attempt can have an empty marking slice. -/
theorem emptyWork_before_last (limit index : Nat) (before : index < Nat.sqrt limit / 2) :
    emptyWork (sieveSize limit) index = 0 := by
  apply emptyWork_of_inside
  have root : oddAt index ≤ Nat.sqrt limit := by unfold oddAt; omega
  have square := Nat.le_sqrt'.mp root
  rw [Nat.pow_two] at square
  unfold sieveSize
  omega

theorem source_mark (env : HeapCalls.Env) (heap : Mutable.Heap) (address index : Nat) (sieve : Array Bool)
    (marker : 1 ≤ index)
    (getIndex : env "i" = some (.scalar (.int index)))
    (getSize : env "size" = some (.scalar (.int sieve.size)))
    (getSieve : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    ∃ finalEnv, BlockContinuesIn .word program env sourceBlock heap
      (6 * writes sieve.size index + emptyWork sieve.size index + 300) finalEnv (heap.set address (.bytearray (sieveBytes (markMultiples (oddAt index) sieve)))) ∧
      Stable env finalEnv := by
  let p := oddAt index
  let start := p * p / 2
  let count := ((sieve.size : Int) - 1 - start) / p + 1
  let e1 := HeapCalls.setLocal env "p" (.scalar (.int p))
  let e2 := HeapCalls.setLocal e1 "start" (.scalar (.int start))
  let e3 := HeapCalls.setLocal e2 "step" (.scalar (.int p))
  let e4 := HeapCalls.setLocal e3 "count" (.scalar (.int count))
  have positive : 0 < p := by unfold p oddAt; omega
  have first : BlockContinuesIn .word program env [sourceBlock[0]'(by decide)] heap 13 e1 heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (av := .int (2 * index : Nat)) (bv := .int 1)
      (ca := 6) (cb := 1) (work := 1)
    · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
        (av := .int 2) (bv := .int index)
      · exact EvaluatesIn.constant
      · exact EvaluatesIn.lookup getIndex
      · simp [Mutable.binary, scalarBinary]
      · rfl
    · exact EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, p, oddAt]
    · rfl
  have second : BlockContinuesIn .word program e1 [sourceBlock[1]'(by decide)] heap 13 e2 heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (av := .int (p * p : Nat)) (bv := .int 2)
      (ca := 6) (cb := 1) (work := 1)
    · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
        (av := .int p) (bv := .int p)
      · exact EvaluatesIn.lookup rfl
      · exact EvaluatesIn.lookup rfl
      · simp [Mutable.binary, scalarBinary]
      · rfl
    · exact EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, start]
    · rfl
  have third : BlockContinuesIn .word program e2 [sourceBlock[2]'(by decide)] heap 3 e3 heap :=
    BlockContinuesIn.assign (EvaluatesIn.lookup (by simp [e2, e1, HeapCalls.setLocal]))
  have fourth : BlockContinuesIn .word program e3 [sourceBlock[3]'(by decide)] heap 23 e4 heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (av := .int (((sieve.size : Int) - 1 - start) / p))
      (bv := .int 1) (ca := 16) (cb := 1) (work := 1)
    · apply EvaluatesIn.binary (av := .int ((sieve.size : Int) - 1 - start))
        (bv := .int p) (ca := 11) (cb := 1) (work := 1)
      · apply EvaluatesIn.binary (ca := 6) (cb := 1) (work := 1)
          (av := .int ((sieve.size : Int) - 1)) (bv := .int start)
        · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
            (av := .int sieve.size) (bv := .int 1)
          · exact EvaluatesIn.lookup (by simpa [e3, e2, e1, HeapCalls.setLocal] using getSize)
          · exact EvaluatesIn.constant
          · rfl
          · rfl
        · exact EvaluatesIn.lookup (by simp [e3, e2, HeapCalls.setLocal])
        · rfl
        · rfl
      · exact EvaluatesIn.lookup (by simp [e3, e2, e1, HeapCalls.setLocal])
      · simp [Mutable.binary, scalarBinary, Nat.ne_of_gt positive, positive]
      · rfl
    · exact EvaluatesIn.constant
    · rfl
    · rfl
  have fifth : BlockContinuesIn .word program e4 [sourceBlock[4]'(by decide)] heap
      (6 * count.toNat + emptyWork sieve.size index + 230) e4 (heap.set address (.bytearray (sieveBytes (markMultiples (oddAt index) sieve)))) := by
    have evaluated : EvaluatesIn .word program e4
        (.binOp (.constant (.bytes [0])) .mult (.name "count")) heap
        (count.toNat + 7) (.scalar (.bytes (List.replicate count.toNat 0))) heap := by
      have run := EvaluatesIn.binary (arithmetic := .word) (module := program)
        (env := e4) (a := .constant (.bytes [0])) (b := .name "count") (op := .mult)
        (heap := heap) (value := .scalar (.bytes (List.replicate count.toNat 0))) (work := count.toNat + 2) EvaluatesIn.constant (EvaluatesIn.lookup rfl)
        (by simp [Mutable.binary, scalarBinary])
        (by simp [PrimitiveWork.binary, scalarNumber, PrimitiveWork.integer])
      exact run.mono (by omega)
    have arguments : ArgumentsIn .word program e4 0
        [.name "is_prime", .name "start", .constant .none, .name "step"] heap 14
        [.ref address, .scalar (.int start), .scalar .none, .scalar (.int p)] heap :=
      .cons (EvaluatesIn.lookup (by simpa [e4, e3, e2, e1, HeapCalls.setLocal] using getSieve))
        (.cons (EvaluatesIn.lookup (by simp [e4, e3, e2, HeapCalls.setLocal]))
          (.cons EvaluatesIn.constant (.cons (EvaluatesIn.lookup
            (by simp [e4, e3, e2, e1, HeapCalls.setLocal])) .nil)))
    have write : Mutable.setSlice (.ref address) (.scalar (.int start)) (.scalar .none) (.scalar (.int p))
        (.scalar (.bytes (List.replicate count.toNat 0))) heap =
        some (.ok (.scalar .none), heap.set address (.bytearray (sieveBytes (markMultiples (oddAt index) sieve)))) := by
      exact markSlice_buffer heap address index sieve marker found
    have price : HeapCalls.BufferWork.slice .word (sieveBytes sieve).length (.scalar (.int start)) (.scalar .none)
        (.scalar (.int p)) (List.replicate count.toNat 0) = some (5 * count.toNat + emptyWork sieve.size index + 205) := by
      have notOne : (p : Int) ≠ 1 := by unfold p oddAt; omega
      have equal : count.toNat = writes sieve.size index := source_count _ _
      simp only [HeapCalls.BufferWork.slice, HeapCalls.BufferWork.sliceWidth, HeapCalls.BufferWork.bound,
        PrimitiveWork.integer, scalarNumber, sieveBytes, List.length_map, Array.length_toList]
      simp [PrimitiveWork.integer, notOne, List.length_replicate, List.isEmpty_replicate, equal, emptyWork]
      split <;> omega
    exact (BlockContinuesIn.buffer_setSlice (lower := some (.name "start")) (upper := none)
      (stride := some (.name "step")) evaluated arguments found write price).mono (by omega)
  refine ⟨e4, (first.append (second.append (third.append (fourth.append fifth)))).mono ?_, ?_⟩
  · have equal : count.toNat = writes sieve.size index := source_count _ _
    omega
  · intro name member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl <;> rfl

end solvers.«7».markingRuntime
