import «.formalsw».solvers.«10.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p10.py»
import ProjectEulerSolutions.Equiv.P10
import FormalSW
import Mathlib.Tactic

namespace solvers.«10»
open FormalSW.PyAst
private abbrev program := Solvers.P10Ast.module

def oddValue (index : Nat) : Nat := 2 * index + 3
def sieveSize (limit : Nat) : Nat := limit / 2 - 1

theorem represented_iff (limit index : Nat) : index < sieveSize limit ↔ oddValue index < limit := by
  unfold sieveSize oddValue
  omega

theorem sieveSize_cast (limit : Nat) (large : 4 ≤ limit) :
    (sieveSize limit : Int) = (limit : Int) / 2 - 1 := by
  unfold sieveSize
  omega

theorem candidate_count (root : Nat) (positive : 0 < root) :
    ((root : Int) - 3) / 2 + 1 = ((root - 1) / 2 : Nat) := by omega

theorem candidate_index (root index : Nat) : index < (root - 1) / 2 ↔ oddValue index ≤ root := by
  unfold oddValue
  omega

def contribution (index : Nat) (byte : UInt8) : Nat := if byte.toNat = 0 then 0 else oddValue index
def oddSum : Nat → List UInt8 → Nat
  | _, [] => 0
  | index, byte :: rest => contribution index byte + oddSum (index + 1) rest

/-- After processing the first candidates, only their composite multiples have been removed. -/
def Survives (processed value : Nat) : Prop :=
  ∀ index, index < processed → Nat.Prime (oddValue index) →
    oddValue index ^ 2 ≤ value → ¬ oddValue index ∣ value

theorem prime_survives (processed value : Nat) (prime : Nat.Prime value) :
    Survives processed value := by
  intro index _ divisorPrime square divides
  have equal := (Nat.dvd_prime prime).mp divides
  have lower : 3 ≤ oddValue index := by unfold oddValue; omega
  rcases equal with one | equal
  · omega
  · nlinarith

theorem odd_minFac (index : Nat) : 3 ≤ (oddValue index).minFac ∧
    (oddValue index).minFac % 2 = 1 := by
  have lower : 3 ≤ oddValue index := by unfold oddValue; omega
  have prime := Nat.minFac_prime (by omega : oddValue index ≠ 1)
  have notTwo : (oddValue index).minFac ≠ 2 := by
    intro equal
    have divides := Nat.minFac_dvd (oddValue index)
    rw [equal] at divides
    have := Nat.mod_eq_zero_of_dvd divides
    unfold oddValue at this
    omega
  exact ⟨by have := prime.two_le; omega, prime.mod_two_eq_one_iff_ne_two.mpr notTwo⟩

theorem survives_current (index : Nat) :
    Survives index (oddValue index) ↔ Nat.Prime (oddValue index) := by
  refine ⟨?_, prime_survives index (oddValue index)⟩
  intro survives
  by_contra composite
  have lower : 3 ≤ oddValue index := by unfold oddValue; omega
  obtain ⟨factorLower, factorOdd⟩ := odd_minFac index
  let factorIndex := ((oddValue index).minFac - 3) / 2
  have factor : oddValue factorIndex = (oddValue index).minFac := by
    change 2 * (((oddValue index).minFac - 3) / 2) + 3 = (oddValue index).minFac
    omega
  have square := Nat.minFac_sq_le_self (by omega : 0 < oddValue index) composite
  have before : factorIndex < index := by
    have smaller : (oddValue index).minFac < oddValue index := by nlinarith
    rw [← factor] at smaller
    unfold oddValue at smaller
    omega
  exact survives factorIndex before (factor ▸ Nat.minFac_prime (by omega))
    (by simpa only [factor] using square) (by simpa only [factor] using Nat.minFac_dvd (oddValue index))

theorem survives_finished (limit index : Nat) (valid : index < sieveSize limit) :
    Survives ((Nat.sqrt (limit - 1) - 1) / 2) (oddValue index) ↔ Nat.Prime (oddValue index) := by
  refine ⟨?_, prime_survives _ _⟩
  intro survives
  by_contra composite
  have lower : 3 ≤ oddValue index := by unfold oddValue; omega
  have upper := (represented_iff limit index).mp valid
  obtain ⟨factorLower, factorOdd⟩ := odd_minFac index
  let factorIndex := ((oddValue index).minFac - 3) / 2
  have factor : oddValue factorIndex = (oddValue index).minFac := by
    change 2 * (((oddValue index).minFac - 3) / 2) + 3 = (oddValue index).minFac
    omega
  have square := Nat.minFac_sq_le_self (by omega : 0 < oddValue index) composite
  have before : factorIndex < (Nat.sqrt (limit - 1) - 1) / 2 := by
    apply (candidate_index _ _).mpr
    rw [factor, Nat.le_sqrt']
    omega
  exact survives factorIndex before (factor ▸ Nat.minFac_prime (by omega))
    (by simpa only [factor] using square) (by simpa only [factor] using Nat.minFac_dvd (oddValue index))

theorem survives_next (processed value : Nat) :
    Survives (processed + 1) value ↔ Survives processed value ∧
      (Nat.Prime (oddValue processed) → oddValue processed ^ 2 ≤ value → ¬ oddValue processed ∣ value) := by
  constructor
  · intro survives
    exact ⟨fun index before => survives index (by omega), survives processed (by omega)⟩
  · rintro ⟨survives, current⟩ index before
    by_cases equal : index = processed
    · simpa only [equal] using current
    · exact survives index (by omega)

def squareIndex (index : Nat) : Nat := 2 * index * index + 6 * index + 3

theorem squareIndex_value (index : Nat) : oddValue (squareIndex index) = oddValue index ^ 2 := by
  unfold oddValue squareIndex
  ring

theorem squareIndex_cast (index : Nat) :
    (squareIndex index : Int) = ((oddValue index : Int) * oddValue index - 3) / 2 := by
  have equal : (oddValue index : Int) * oddValue index - 3 = 2 * squareIndex index := by
    simp only [squareIndex, oddValue, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
    ring
  rw [equal]
  omega

def markedIndices (length index : Nat) : List Nat :=
  List.range' (squareIndex index)
    ((length - squareIndex index + oddValue index - 1) / oddValue index) (oddValue index)

theorem mem_markedIndices (length candidate index : Nat) :
    index ∈ markedIndices length candidate ↔ index < length ∧
      oddValue candidate ^ 2 ≤ oddValue index ∧ oddValue candidate ∣ oddValue index := by
  have positive : 0 < oddValue candidate := by unfold oddValue; omega
  rw [markedIndices, FormalSW.Py.Builtin.Slice.mem_positive_end _ _ _ _ positive]
  have bound : squareIndex candidate ≤ index ↔ oddValue candidate ^ 2 ≤ oddValue index := by
    rw [← squareIndex_value]
    unfold oddValue
    omega
  have divisible (lower : squareIndex candidate ≤ index) :
      oddValue candidate ∣ index - squareIndex candidate ↔ oddValue candidate ∣ oddValue index := by
    have coprime : (oddValue candidate).Coprime 2 := by
      apply Nat.coprime_two_right.mpr
      exact ⟨candidate + 1, by unfold oddValue; omega⟩
    have equal : oddValue index = oddValue candidate ^ 2 + 2 * (index - squareIndex candidate) := by
      rw [← squareIndex_value]
      unfold oddValue
      omega
    rw [equal, ← Nat.dvd_add_iff_right (dvd_pow_self _ (by decide : 2 ≠ 0)), coprime.dvd_mul_left]
  constructor
  · rintro ⟨lower, upper, divides⟩
    exact ⟨upper, bound.mp lower, (divisible lower).mp divides⟩
  · rintro ⟨upper, square, divides⟩
    exact ⟨bound.mpr square, upper, (divisible (bound.mpr square)).mpr divides⟩

def mark (bytes : List UInt8) (candidate : Nat) : List UInt8 :=
  (markedIndices bytes.length candidate).foldl (fun result index => result.set index 0) bytes

theorem mark_length (bytes : List UInt8) (candidate : Nat) : (mark bytes candidate).length = bytes.length := by
  unfold mark
  generalize markedIndices bytes.length candidate = indices
  induction indices generalizing bytes with
  | nil => rfl
  | cons index rest ih => simpa only [List.foldl_cons, List.length_set] using ih (bytes.set index 0)

theorem mark_get (bytes : List UInt8) (candidate index : Nat) (valid : index < bytes.length) :
    (mark bytes candidate)[index]? =
      if oddValue candidate ^ 2 ≤ oddValue index ∧ oddValue candidate ∣ oddValue index
      then some 0 else bytes[index]? := by
  rw [mark, FormalSW.Py.Builtin.List.getElem?_setUniform _ _ _ _ valid]
  simp only [mem_markedIndices, valid, true_and]

def Sieved (limit processed : Nat) (bytes : List UInt8) : Prop :=
  bytes.length = sieveSize limit ∧ ∀ index byte, bytes[index]? = some byte →
    (byte.toNat ≠ 0 ↔ Survives processed (oddValue index))

theorem sieved_initial (limit : Nat) : Sieved limit 0 (List.replicate (sieveSize limit) 1) := by
  refine ⟨by simp, ?_⟩
  intro index byte found
  have equal : byte = 1 := List.eq_of_mem_replicate (List.mem_of_getElem? found)
  subst byte
  simp [Survives]

theorem sieved_skip (limit processed : Nat) (bytes : List UInt8)
    (sieved : Sieved limit processed bytes) (composite : ¬ Nat.Prime (oddValue processed)) :
    Sieved limit (processed + 1) bytes := by
  refine ⟨sieved.1, ?_⟩
  intro index byte found
  rw [sieved.2 index byte found, survives_next]
  simp [composite]

theorem sieved_mark (limit processed : Nat) (bytes : List UInt8)
    (sieved : Sieved limit processed bytes) (prime : Nat.Prime (oddValue processed)) :
    Sieved limit (processed + 1) (mark bytes processed) := by
  refine ⟨by rw [mark_length, sieved.1], ?_⟩
  intro index byte found
  have valid : index < bytes.length := by
    have := List.getElem?_eq_some_iff.mp found
    simpa only [mark_length] using this.1
  rw [mark_get bytes processed index valid] at found
  rw [survives_next]
  by_cases removed : oddValue processed ^ 2 ≤ oddValue index ∧ oddValue processed ∣ oddValue index
  · simp only [if_pos removed, Option.some.injEq] at found
    subst byte
    simp [prime, removed.1, removed.2]
  · simp only [if_neg removed] at found
    rw [sieved.2 index byte found]
    simp only [prime, true_implies]
    tauto

theorem mark_empty (bytes : List UInt8) (processed : Nat) (outside : bytes.length ≤ squareIndex processed) :
    mark bytes processed = bytes := by
  have empty : markedIndices bytes.length processed = [] := by
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro index selected
    obtain ⟨valid, square, _⟩ := (mem_markedIndices _ _ _).mp selected
    rw [← squareIndex_value] at square
    unfold oddValue at square
    omega
  simp only [mark, empty, List.foldl_nil]

theorem mark_slice (bytes : List UInt8) (processed : Nat) (inside : squareIndex processed < bytes.length) :
    FormalSW.Py.Builtin.ByteArray.setSlice bytes
      { start := some (squareIndex processed), step := some (oddValue processed) }
      (List.replicate (((bytes.length : Int) - squareIndex processed - 1) / oddValue processed + 1).toNat 0) =
      .ok (mark bytes processed) := by
  have positive : 0 < oddValue processed := by unfold oddValue; omega
  have normalized : FormalSW.Py.Builtin.Slice.indices
      { start := some (squareIndex processed), step := some (oddValue processed) } bytes.length =
      .ok ((squareIndex processed : Int), (bytes.length : Int), (oddValue processed : Int)) := by
    have nonzero : (oddValue processed : Int) ≠ 0 := by omega
    have nonnegative : ¬ (squareIndex processed : Int) < 0 := by omega
    simp only [FormalSW.Py.Builtin.Slice.indices, Option.getD_some, nonzero, ↓reduceIte,
      show (0 : Int) < oddValue processed by omega]
    change Except.ok (max 0 (min (if (squareIndex processed : Int) < 0
      then (squareIndex processed : Int) + bytes.length else squareIndex processed) bytes.length),
      (bytes.length : Int), (oddValue processed : Int)) = _
    simp only [if_neg nonnegative]
    congr 2
    omega
  have selected := FormalSW.Py.Builtin.Slice.indexList_positive_end bytes.length
    (squareIndex processed) (oddValue processed) positive
  have count : (((bytes.length : Int) - squareIndex processed - 1) / oddValue processed + 1).toNat =
      (markedIndices bytes.length processed).length := by
    rw [markedIndices, List.length_range']
    rw [show (bytes.length : Int) - squareIndex processed - 1 =
      (bytes.length : Int) - 1 - squareIndex processed by omega]
    exact FormalSW.Py.Builtin.Slice.positive_end_count _ _ _ positive
  rw [count]
  exact FormalSW.Py.Builtin.ByteArray.setSlice_uniform bytes _ 0 normalized
    (by unfold oddValue; omega) selected

def oddPrimeSum : Nat → Nat
  | 0 => 0
  | count + 1 => oddPrimeSum count + if Nat.Prime (oddValue count) then oddValue count else 0

theorem naive_odd (count : Nat) : ProjectEulerStatements.P10.naive (2 * count + 3) = 2 + oddPrimeSum count := by
  induction count with
  | zero => exact ProjectEulerSolutions.P10.naive_three
  | succ count ih =>
      have evenComposite : ¬ Nat.Prime (2 * count + 4) := by
        apply ProjectEulerSolutions.P10.not_prime_even_gt_two (by omega)
        exact ⟨count + 2, by omega⟩
      rw [show 2 * (count + 1) + 3 = (2 * count + 3 + 1) + 1 by omega,
        ProjectEulerSolutions.P10.naive_succ, ProjectEulerSolutions.P10.naive_succ, ih]
      simp [evenComposite, oddPrimeSum, oddValue, Nat.add_assoc]

theorem naive_sieveSize (limit : Nat) (large : 3 ≤ limit) :
    ProjectEulerStatements.P10.naive limit = 2 + oddPrimeSum (sieveSize limit) := by
  have bound : limit = 2 * sieveSize limit + 2 ∨ limit = 2 * sieveSize limit + 3 := by
    unfold sieveSize
    omega
  rcases bound with even | odd
  · have composite : ¬ Nat.Prime limit := by
      apply ProjectEulerSolutions.P10.not_prime_even_gt_two (by omega)
      exact ⟨sieveSize limit + 1, by omega⟩
    have next := naive_odd (sieveSize limit)
    rw [show 2 * sieveSize limit + 3 = limit + 1 by omega,
      ProjectEulerSolutions.P10.naive_succ, if_neg composite, Nat.add_zero] at next
    exact next
  · rw [odd]
    -- Keep the size from before the rewrite rather than normalize a nested quotient.
    have size : sieveSize (2 * sieveSize limit + 3) = sieveSize limit := by unfold sieveSize; omega
    rw [size, naive_odd]

theorem oddSum_primes (bytes : List UInt8) (position : Nat)
    (correct : ∀ index byte, bytes[index]? = some byte →
      (byte.toNat ≠ 0 ↔ Nat.Prime (oddValue (position + index)))) :
    oddPrimeSum position + oddSum position bytes = oddPrimeSum (position + bytes.length) := by
  induction bytes generalizing position with
  | nil => simp [oddSum]
  | cons byte rest ih =>
      have head := correct 0 byte (by simp)
      have contribution : contribution position byte =
          if Nat.Prime (oddValue position) then oddValue position else 0 := by
        simp only [Nat.add_zero] at head
        unfold contribution
        by_cases zero : byte.toNat = 0 <;> simp_all
      have tail := ih (position + 1) (by
        intro index value found
        simpa only [Nat.add_assoc, Nat.add_comm 1 index] using correct (index + 1) value found)
      simpa only [oddPrimeSum, oddSum, contribution, Nat.add_assoc, Nat.add_left_comm,
        Nat.add_comm, List.length_cons] using tail

theorem sieved_sum (limit : Nat) (large : 3 ≤ limit) (bytes : List UInt8)
    (sieved : Sieved limit ((Nat.sqrt (limit - 1) - 1) / 2) bytes) :
    2 + oddSum 0 bytes = ProjectEulerStatements.P10.naive limit := by
  have sum := oddSum_primes bytes 0 (by
    intro index byte found
    have valid : index < sieveSize limit := by
      rw [← sieved.1]
      exact (List.getElem?_eq_some_iff.mp found).1
    simpa only [Nat.zero_add] using (sieved.2 index byte found).trans (survives_finished limit index valid))
  simpa only [oddPrimeSum, Nat.zero_add, sieved.1, naive_sieveSize limit large] using congrArg (2 + ·) sum

private abbrev sumExpression : Expr :=
  .binOp (.binOp (.constant (.int 2)) .mult (.name "i")) .add (.constant (.int 3))
private abbrev sumBody : List Stmt :=
  [.ifThen (.name "is_prime") [.augAssign (.name "total") .add sumExpression] []]

private def solveBody : List Stmt :=
  match findFunctionIn program.body "solve" with
  | some (_, body) => executableBody body
  | none => []

private def sieveIteration : List Stmt :=
  match (solveBody[6]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private def markBody : List Stmt :=
  match sieveIteration with
  | [.ifThen _ body _] => body
  | _ => []

private def writeBody : List Stmt :=
  match (markBody[2]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

private def markEnv (env : HeapCalls.Env) (index size : Nat) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal env
    "p" (.scalar (.int (oddValue index)))) "start" (.scalar (.int (squareIndex index))))
    "count" (.scalar (.int (((size : Int) - squareIndex index - 1) / oddValue index + 1)))

/-- The actual assignments and extended-slice write preserve the sieve's address. -/
theorem mark.result (env : HeapCalls.Env) (heap : Mutable.Heap) (address index : Nat) (bytes : List UInt8)
    (inside : squareIndex index < bytes.length)
    (getIndex : env "i" = some (.scalar (.int index)))
    (getSize : env "size" = some (.scalar (.int bytes.length)))
    (getSieve : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    HeapCalls.BlockContinues program env markBody heap (markEnv env index bytes.length)
      (heap.set address (.bytearray (mark bytes index))) := by
  let p := oddValue index
  let start := squareIndex index
  let count := ((bytes.length : Int) - start - 1) / p + 1
  let e1 := HeapCalls.setLocal env "p" (.scalar (.int p))
  let e2 := HeapCalls.setLocal e1 "start" (.scalar (.int start))
  let e3 := HeapCalls.setLocal e2 "count" (.scalar (.int count))
  have positive : 0 < p := by unfold p oddValue; omega
  have first : HeapCalls.BlockContinues program env [markBody[0]'(by decide)] heap e1 heap := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (av := .scalar (.int (2 * index : Nat)))
      (bv := .scalar (.int 3)) (middle := heap) (afterRight := heap)
      (HeapCalls.Evaluates.binary HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup getIndex)
        (by simp [Mutable.binary, scalarBinary])) HeapCalls.Evaluates.constant
    simp [Mutable.binary, scalarBinary, p, oddValue]
  have second : HeapCalls.BlockContinues program e1 [markBody[1]'(by decide)] heap e2 heap := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (av := .scalar (.int ((p : Int) * p - 3)))
      (bv := .scalar (.int 2)) (middle := heap) (afterRight := heap)
    · exact HeapCalls.Evaluates.binary
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) (HeapCalls.Evaluates.lookup rfl) rfl)
        HeapCalls.Evaluates.constant rfl
    · exact HeapCalls.Evaluates.constant
    · change (some (.ok (.scalar (.int (((oddValue index : Int) * oddValue index - 3) / 2))), heap) :
        Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
      rw [← squareIndex_cast]
  have third : HeapCalls.BlockContinues program e2 [writeBody[0]'(by decide)] heap e3 heap := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (av := .scalar (.int (((bytes.length : Int) - start - 1) / p)))
      (bv := .scalar (.int 1)) (middle := heap) (afterRight := heap)
    · apply HeapCalls.Evaluates.binary (av := .scalar (.int ((bytes.length : Int) - start - 1)))
        (bv := .scalar (.int p)) (middle := heap) (afterRight := heap)
      · exact HeapCalls.Evaluates.binary
          (HeapCalls.Evaluates.binary
            (HeapCalls.Evaluates.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getSize))
            (HeapCalls.Evaluates.lookup rfl) rfl) HeapCalls.Evaluates.constant rfl
      · exact HeapCalls.Evaluates.lookup (by simp [e2, e1, HeapCalls.setLocal])
      · simp [Mutable.binary, scalarBinary, Nat.ne_of_gt positive, positive]
    · exact HeapCalls.Evaluates.constant
    · rfl
  have fourth : HeapCalls.BlockContinues program e3 [writeBody[1]'(by decide)] heap e3
      (heap.set address (.bytearray (mark bytes index))) := by
    apply HeapCalls.BlockContinues.setSlice
      (value := .scalar (.bytes (List.replicate count.toNat 0))) (afterRhs := heap) (afterTarget := heap)
      (receiver := .ref address) (lowerValue := .scalar (.int start))
      (upperValue := .scalar .none) (strideValue := .scalar (.int p))
    · exact HeapCalls.Evaluates.binary HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup rfl)
        (by simp [Mutable.binary, scalarBinary])
    · exact .cons (HeapCalls.Evaluates.lookup (by simpa [e3, e2, e1, HeapCalls.setLocal] using getSieve))
        (.cons (HeapCalls.Evaluates.lookup (by simp [e3, e2, HeapCalls.setLocal]))
          (.cons HeapCalls.Evaluates.constant (.cons (HeapCalls.Evaluates.lookup
            (by simp [e3, e2, e1, HeapCalls.setLocal])) .nil)))
    · apply Mutable.setSlice_buffer heap address bytes _ _ _ _ _
        { start := some start, step := some p } found
      · change (if (p : Int) = 0 then some (.error ⟨"ValueError", []⟩, heap)
          else some (.ok { start := some (start : Int), stop := none, step := some (p : Int) }, heap) :
            Option (Except ExceptionObject FormalSW.Py.Builtin.Slice.Slice × Mutable.Heap)) = _
        rw [if_neg (by omega)]
      · exact mark_slice bytes index inside
  apply first.append (second.append ?_)
  apply HeapCalls.BlockContinues.branch (flag := true) (value := .scalar (.bool true)) (middle := heap)
  · apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl)
      (HeapCalls.Evaluates.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getSize))
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, start, inside]
  · rfl
  · exact third.append fourth

private def SieveStable (before after : HeapCalls.Env) : Prop :=
  ∀ name ∈ ["size", "sieve"], after name = before name

private theorem SieveStable.refl (env : HeapCalls.Env) : SieveStable env env := by
  intro name member
  rfl

private theorem SieveStable.trans {first middle last : HeapCalls.Env}
    (a : SieveStable first middle) (b : SieveStable middle last) : SieveStable first last := by
  intro name member
  exact (b name member).trans (a name member)

private theorem SieveStable.index (env : HeapCalls.Env) (index : Nat) :
    SieveStable env (HeapCalls.setLocal env "i" (.scalar (.int index))) := by
  intro name member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> rfl

private theorem SieveStable.mark (env : HeapCalls.Env) (index size : Nat) :
    SieveStable env (markEnv env index size) := by
  intro name member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> rfl

theorem sieve.iteration (limit index : Nat) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (valid : index < bytes.length) (inside : squareIndex index < bytes.length)
    (getIndex : env "i" = some (.scalar (.int index)))
    (getSize : env "size" = some (.scalar (.int bytes.length)))
    (getSieve : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (sieved : Sieved limit index bytes) :
    ∃ finalEnv finalHeap finalBytes,
      HeapCalls.BlockContinues program env sieveIteration heap finalEnv finalHeap ∧
      finalHeap[address]? = some (.bytearray finalBytes) ∧
      Sieved limit (index + 1) finalBytes ∧ SieveStable env finalEnv := by
  have read : HeapCalls.Evaluates program env (.subscript (.name "sieve") (.name "i")) heap
      (.scalar (.int bytes[index].toNat)) heap := by
    apply HeapCalls.Evaluates.subscript (by intros a b c impossible; cases impossible)
      (HeapCalls.Evaluates.lookup getSieve) (HeapCalls.Evaluates.lookup getIndex)
    exact Mutable.getItem_buffer heap address index bytes found valid
  have prime : bytes[index].toNat ≠ 0 ↔ Nat.Prime (oddValue index) :=
    (sieved.2 index bytes[index] (List.getElem?_eq_getElem valid)).trans (survives_current index)
  by_cases zero : bytes[index].toNat = 0
  · refine ⟨env, heap, bytes, ?_, found, sieved_skip limit index bytes sieved ?_, SieveStable.refl env⟩
    · apply HeapCalls.BlockContinues.branch (flag := false) read
      · simp [Mutable.truth, scalarTruth, zero]
      · exact HeapCalls.BlockContinues.nil
    · simpa only [zero, ne_eq, not_true_eq_false, false_iff] using prime
  · refine ⟨markEnv env index bytes.length, heap.set address (.bytearray (mark bytes index)),
      mark bytes index, ?_, ?_, sieved_mark limit index bytes sieved (prime.mp zero), SieveStable.mark _ _ _⟩
    · apply HeapCalls.BlockContinues.branch (flag := true) read
      · simp [Mutable.truth, scalarTruth, zero]
      · exact mark.result env heap address index bytes inside getIndex getSize getSieve found
    · exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp found).1

theorem sieve.loop (limit next : Nat) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (bounded : next ≤ (Nat.sqrt (limit - 1) - 1) / 2)
    (getSize : env "size" = some (.scalar (.int bytes.length)))
    (getSieve : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (sieved : Sieved limit next bytes) :
    ∃ finalEnv finalHeap finalBytes,
      HeapCalls.BlockContinues program env
        [.rangeStepLoop "i" next ((Nat.sqrt (limit - 1) - 1) / 2 : Nat) 1 sieveIteration []]
        heap finalEnv finalHeap ∧
      finalHeap[address]? = some (.bytearray finalBytes) ∧
      Sieved limit ((Nat.sqrt (limit - 1) - 1) / 2) finalBytes ∧ SieveStable env finalEnv := by
  by_cases active : next < (Nat.sqrt (limit - 1) - 1) / 2
  · have rootBound := (candidate_index (Nat.sqrt (limit - 1)) next).mp active
    have square := Nat.le_sqrt'.mp rootBound
    have rootBelow := Nat.sqrt_le_self (limit - 1)
    have lower : 3 ≤ oddValue next := by unfold oddValue; omega
    have large : 0 < limit := by omega
    have valid : next < bytes.length := by
      rw [sieved.1, represented_iff]
      omega
    have inside : squareIndex next < bytes.length := by
      rw [sieved.1, represented_iff, squareIndex_value]
      omega
    let indexed := HeapCalls.setLocal env "i" (.scalar (.int next))
    have indexedStable : SieveStable env indexed := SieveStable.index env next
    obtain ⟨middleEnv, middleHeap, middleBytes, iteration, middleFound, middleSieved, stable⟩ :=
      sieve.iteration limit next bytes indexed heap address valid inside rfl
        ((indexedStable "size" (by simp)).trans getSize)
        ((indexedStable "sieve" (by simp)).trans getSieve) found sieved
    have middleSize : middleBytes.length = bytes.length := middleSieved.1.trans sieved.1.symm
    obtain ⟨finalEnv, finalHeap, finalBytes, rest, finalFound, finalSieved, finalStable⟩ :=
      sieve.loop limit (next + 1) middleBytes middleEnv middleHeap address (by omega)
        (by rw [middleSize]; exact (stable "size" (by simp)).trans ((indexedStable "size" (by simp)).trans getSize))
        ((stable "sieve" (by simp)).trans ((indexedStable "sieve" (by simp)).trans getSieve))
        middleFound middleSieved
    refine ⟨finalEnv, finalHeap, finalBytes, ?_, finalFound, finalSieved,
      indexedStable.trans (stable.trans finalStable)⟩
    apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
    simpa only [Nat.cast_add, Nat.cast_one] using iteration.append rest
  · have equal : next = (Nat.sqrt (limit - 1) - 1) / 2 := by omega
    refine ⟨env, heap, bytes, ?_, found, equal ▸ sieved, SieveStable.refl env⟩
    exact HeapCalls.BlockContinues.range_done (by simp [Calls.rangeContinues]; omega)
      HeapCalls.BlockContinues.nil
termination_by (Nat.sqrt (limit - 1) - 1) / 2 - next
decreasing_by omega

/-- The final pass reads the shared sieve and adds exactly the represented nonzero entries. -/
theorem sumLoop.result (heap : Mutable.Heap) (address : Nat) (bytes : List UInt8)
    (found : heap[address]? = some (.bytearray bytes)) (position : Nat) (bounded : position ≤ bytes.length)
    (total : Int) (env : HeapCalls.Env) (getTotal : env "total" = some (.scalar (.int total))) :
    ∃ finalEnv, HeapCalls.BlockContinues program env
      [.enumerateStepLoop "i" "is_prime" (.ref address) position position sumBody []] heap finalEnv heap ∧
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
    have step : ∃ middleEnv, HeapCalls.BlockContinues program boundEnv sumBody heap middleEnv heap ∧
        middleEnv "total" = some (.scalar (.int (total + contribution position byte))) := by
      by_cases zero : byte.toNat = 0
      · refine ⟨boundEnv, ?_, ?_⟩
        · apply HeapCalls.BlockContinues.branch (flag := false) (HeapCalls.Evaluates.lookup rfl)
          · simp [Mutable.truth, scalarTruth, zero]
          · exact HeapCalls.BlockContinues.nil
        · simpa [contribution, zero] using saved
      · refine ⟨HeapCalls.setLocal boundEnv "total" (.scalar (.int (total + contribution position byte))), ?_, rfl⟩
        apply HeapCalls.BlockContinues.branch (flag := true) (HeapCalls.Evaluates.lookup rfl)
        · simp [Mutable.truth, scalarTruth, zero]
        · have value : HeapCalls.Evaluates program boundEnv sumExpression heap
              (.scalar (.int (oddValue position))) heap := by
            apply HeapCalls.Evaluates.binary
              (HeapCalls.Evaluates.binary HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup (name := "i") rfl) rfl)
              HeapCalls.Evaluates.constant
            simp [Mutable.binary, scalarBinary, oddValue]
          apply HeapCalls.BlockContinues.augAssign saved value
          simp [HeapCalls.inplace, Mutable.binary, scalarBinary, contribution, zero]
    obtain ⟨middleEnv, step, savedNext⟩ := step
    obtain ⟨finalEnv, rest, answer⟩ := sumLoop.result heap address bytes found (position + 1) (by omega)
      (total + contribution position byte) middleEnv savedNext
    refine ⟨finalEnv, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.enumerate_step (source := .ref address)
        (value := .scalar (.int byte.toNat)) sequence
        (by simp [List.getElem?_map, List.getElem?_eq_getElem active, byte])
      simpa only [Nat.cast_add, Nat.cast_one] using HeapCalls.BlockContinues.append step rest
    · have tail : bytes.drop position = byte :: bytes.drop (position + 1) := List.drop_eq_getElem_cons active
      simpa only [tail, oddSum, Nat.cast_add, Int.add_assoc] using answer
  · have equal : position = bytes.length := by omega
    refine ⟨env, HeapCalls.BlockContinues.enumerate_done (source := .ref address) sequence ?_ HeapCalls.BlockContinues.nil, ?_⟩
    · simp [equal]
    · simpa [equal, oddSum] using getTotal
termination_by bytes.length - position
decreasing_by omega

private def setupEnv (env : HeapCalls.Env) (address limit : Nat) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal env
    "size" (.scalar (.int (sieveSize limit)))) "sieve" (.ref address))
    "r" (.scalar (.int (Nat.sqrt (limit - 1)))))
    "max_i" (.scalar (.int (((Nat.sqrt (limit - 1) : Int) - 3) / 2)))

private def setupHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1]] ++ [.bytearray (List.replicate (sieveSize limit) 1)]

theorem sieve.initialize (limit : Nat) (large : 4 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (.scalar (.int limit))) :
    HeapCalls.BlockContinues program env ((solveBody.drop 2).take 4) heap
      (setupEnv env (heap.length + 1) limit) (setupHeap heap limit) := by
  let e1 := HeapCalls.setLocal env "size" (.scalar (.int (sieveSize limit)))
  let e2 := HeapCalls.setLocal e1 "sieve" (.ref (heap.length + 1))
  let e3 := HeapCalls.setLocal e2 "r" (.scalar (.int (Nat.sqrt (limit - 1))))
  let e4 := setupEnv env (heap.length + 1) limit
  let heap2 := setupHeap heap limit
  have first : HeapCalls.BlockContinues program env [solveBody[2]'(by decide)] heap e1 heap := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (av := .scalar (.int (limit / 2 : Nat)))
      (bv := .scalar (.int 1)) (middle := heap) (afterRight := heap)
    · exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getLimit)
        HeapCalls.Evaluates.constant (by simp [Mutable.binary, scalarBinary])
    · exact HeapCalls.Evaluates.constant
    · simp [Mutable.binary, scalarBinary, sieveSize_cast limit large]
  have second : HeapCalls.BlockContinues program e1 [solveBody[3]'(by decide)] heap e2 heap2 := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (av := .ref heap.length) (middle := heap ++ [.bytearray [1]])
      (bv := .scalar (.int (sieveSize limit))) (afterRight := heap ++ [.bytearray [1]])
    · apply HeapCalls.Evaluates.builtin (.cons HeapCalls.Evaluates.constant .nil)
      rfl
    · exact HeapCalls.Evaluates.lookup rfl
    · simp [heap2, setupHeap, Mutable.binary, Mutable.allocateBuffer, Except.map,
        show (BinOp.mult == .mult) = true from rfl]
  have third : HeapCalls.BlockContinues program e2 [solveBody[4]'(by decide)] heap2 e3 heap2 := by
    apply HeapCalls.BlockContinues.assign
    have argument : HeapCalls.Evaluates program e2
        (.binOp (.name "limit") .sub (.constant (.int 1))) heap2
        (.scalar (.int (limit - 1 : Nat))) heap2 := by
      apply HeapCalls.Evaluates.binary
        (HeapCalls.Evaluates.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getLimit))
        HeapCalls.Evaluates.constant
      simp [Mutable.binary, scalarBinary, Int.natCast_sub (show 1 ≤ limit by omega)]
    apply HeapCalls.Evaluates.qualified_call (by decide) (.cons argument .nil)
    intro anchors stack
    apply HeapCalls.Reaches.next (following := ⟨.value (.scalar (.int (Nat.sqrt (limit - 1)))), stack, heap2⟩)
    · change HeapCalls.result anchors heap2
        (HeapCalls.scalarResult (some (Math.isqrtCall [.int (limit - 1 : Nat)]))) stack = _
      have root : Math.isqrt (limit - 1) = Nat.sqrt (limit - 1) := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
      simp [HeapCalls.result, HeapCalls.run, HeapCalls.scalarResult, Math.isqrtCall, root,
        show ¬ ((limit - 1 : Nat) : Int) < 0 by omega]
    · exact .refl
  have fourth : HeapCalls.BlockContinues program e3 [solveBody[5]'(by decide)] heap2 e4 heap2 := by
    apply HeapCalls.BlockContinues.assign
    exact HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant rfl)
      HeapCalls.Evaluates.constant rfl
  exact first.append (second.append (third.append fourth))

theorem sieve.prepare (limit : Nat) (large : 4 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (.scalar (.int limit))) :
    ∃ finalEnv finalHeap finalBytes,
      HeapCalls.BlockContinues program env ((solveBody.drop 2).take 5) heap finalEnv finalHeap ∧
      finalEnv "sieve" = some (.ref (heap.length + 1)) ∧
      finalHeap[heap.length + 1]? = some (.bytearray finalBytes) ∧
      Sieved limit ((Nat.sqrt (limit - 1) - 1) / 2) finalBytes := by
  let initialized := setupEnv env (heap.length + 1) limit
  let heap2 := setupHeap heap limit
  have getSize : initialized "size" = some (.scalar (.int (List.replicate (sieveSize limit) (1 : UInt8)).length)) := by
    simp [initialized, setupEnv, HeapCalls.setLocal]
  have getSieve : initialized "sieve" = some (.ref (heap.length + 1)) := rfl
  have found : heap2[heap.length + 1]? = some (.bytearray (List.replicate (sieveSize limit) 1)) := by
    simp [heap2, setupHeap]
  obtain ⟨finalEnv, finalHeap, finalBytes, loop, finalFound, finalSieved, stable⟩ :=
    sieve.loop limit 0 (List.replicate (sieveSize limit) 1) initialized heap2 (heap.length + 1)
      (by omega) getSize getSieve found (sieved_initial limit)
  have ranged : HeapCalls.BlockContinues program initialized [solveBody[6]'(by decide)]
      heap2 finalEnv finalHeap := by
    apply HeapCalls.BlockContinues.for_range
      (values := [.scalar (.int ((Nat.sqrt (limit - 1) - 1) / 2 : Nat))])
      (middle := heap2) (start := 0) (stop := ((Nat.sqrt (limit - 1) - 1) / 2 : Nat)) (stride := 1)
    · apply HeapCalls.Evaluates.Arguments.cons
      · apply HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        have count := candidate_count (Nat.sqrt (limit - 1)) (Nat.sqrt_pos.mpr (by omega))
        simpa only [count] using (show Mutable.binary .add
          (.scalar (.int (((Nat.sqrt (limit - 1) : Int) - 3) / 2))) (.scalar (.int 1)) heap2 =
          some (.ok (.scalar (.int (((Nat.sqrt (limit - 1) : Int) - 3) / 2 + 1))), heap2) from rfl)
      · exact .nil
    · rfl
    · exact loop
  exact ⟨finalEnv, finalHeap, finalBytes, (sieve.initialize limit large env heap getLimit).append ranged,
    (stable "sieve" (by simp)).trans getSieve, finalFound, finalSieved⟩

theorem sum.result (limit : Nat) (large : 3 ≤ limit) (bytes : List UInt8)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (getSieve : env "sieve" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes))
    (sieved : Sieved limit ((Nat.sqrt (limit - 1) - 1) / 2) bytes) :
    HeapCalls.BlockReturns program env (solveBody.drop 7) heap
      (.scalar (.int (ProjectEulerStatements.P10.naive limit))) heap := by
  let initialized := HeapCalls.setLocal env "total" (.scalar (.int 2))
  obtain ⟨finalEnv, loop, answer⟩ := sumLoop.result heap address bytes found 0 (by omega) 2 initialized rfl
  have result : finalEnv "total" = some (.scalar (.int (ProjectEulerStatements.P10.naive limit))) := by
    have mathematical := sieved_sum limit large bytes sieved
    rw [← mathematical]
    simpa using answer
  apply HeapCalls.BlockReturns.assign (value := .scalar (.int 2)) HeapCalls.Evaluates.constant
  apply HeapCalls.BlockContinues.returns (body := [solveBody[8]'(by decide)]) (middleEnv := finalEnv)
  · apply HeapCalls.BlockContinues.for_enumerate
      (items := bytes.map (fun byte => .scalar (.int byte.toNat)))
      (HeapCalls.Evaluates.lookup (by simpa [initialized, HeapCalls.setLocal] using getSieve))
    · simp [Mutable.sequence, found]
    · exact loop
  · exact HeapCalls.BlockReturns.return_value (HeapCalls.Evaluates.lookup result)

/-- The source returns the mathematical prime sum for every nonnegative limit. -/
theorem solve.naturals (limit : Nat) (heap : Mutable.Heap) :
    ∃ finalHeap, HeapCalls.ReturnsValue program "solve" [.scalar (.int limit)] heap
      (.scalar (.int (ProjectEulerStatements.P10.naive limit))) finalHeap := by
  let env := HeapCalls.argumentEnv [⟨"limit", some (.name "int"), none⟩] [.scalar (.int limit)]
  have guard (bound : Nat) : HeapCalls.Evaluates program env
      (.compare (.name "limit") [(.ltE, .constant (.int bound))]) heap
      (.scalar (.bool (decide (limit ≤ bound)))) heap := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases small : limit ≤ 2
  · have zero : ProjectEulerStatements.P10.naive limit = 0 := by
      rw [ProjectEulerSolutions.P10.equiv_of_le_two small, ProjectEulerSolutions.P10.solve_of_le_two small]
    refine ⟨heap, ?_⟩
    rw [zero]
    apply HeapCalls.BlockReturns.function rfl rfl
    apply HeapCalls.BlockReturns.branch (flag := true) (value := .scalar (.bool true)) (middle := heap) (by simpa [small] using guard 2)
    · rfl
    · exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  · by_cases three : limit ≤ 3
    · have equal : limit = 3 := by omega
      refine ⟨heap, ?_⟩
      apply HeapCalls.BlockReturns.function rfl rfl
      apply HeapCalls.BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap) (by simpa [small] using guard 2)
      · rfl
      · apply HeapCalls.BlockReturns.branch (flag := true) (value := .scalar (.bool true)) (middle := heap) (by simpa [three] using guard 3)
        · rfl
        · rw [equal, ProjectEulerSolutions.P10.naive_three]
          exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
    · obtain ⟨finalEnv, finalHeap, bytes, prepare, getSieve, found, sieved⟩ :=
        sieve.prepare limit (by omega) env heap rfl
      refine ⟨finalHeap, ?_⟩
      apply HeapCalls.BlockReturns.function rfl rfl
      apply HeapCalls.BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap) (by simpa [small] using guard 2)
      · rfl
      · apply HeapCalls.BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap) (by simpa [three] using guard 3)
        · rfl
        · exact prepare.returns (sum.result limit (by omega) bytes finalEnv finalHeap (heap.length + 1)
            getSieve found sieved)

theorem solve.«Equiv(naive)» (limit : Int) (domain : Calls.Condition program
    (fun name => if name = "limit" then some (.int limit) else none) Solvers.P10Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (Calls.function P10Ast.module "naive" [.int limit]) := by
  have nonnegative : 0 ≤ limit := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare domain rfl
  lift limit to Nat using nonnegative
  obtain ⟨heap, executed⟩ := solve.naturals limit []
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := [.int limit]) executed
  have reference := (data.«project-euler-statements».data.py.project_euler_statements.p10.naive.result limit).finishes
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem solve.«Supported» :
    (∀ limit : Int, Calls.Condition program
      (fun name => if name = "limit" then some (.int limit) else none) Solvers.P10Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, Calls.Condition program
      (fun name => if name = "limit" then some (.int limit) else none) Solvers.P10Ast.requires_solve_0 →
      (Calls.function P10Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit domain
    have nonnegative : 0 ≤ limit := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare domain rfl
    lift limit to Nat using nonnegative
    obtain ⟨heap, executed⟩ := solve.naturals limit []
    exact (HeapCalls.ReturnsValue.scalar_returns (values := [.int limit]) executed).supported
  · intro limit domain
    have nonnegative : 0 ≤ limit := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare domain rfl
    lift limit to Nat using nonnegative
    exact (data.«project-euler-statements».data.py.project_euler_statements.p10.naive.result limit).returns.supported

end solvers.«10»
