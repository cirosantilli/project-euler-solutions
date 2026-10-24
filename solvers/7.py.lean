import «.formalsw».solvers.«7.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p7.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P7
import Mathlib.Tactic

namespace ProjectEulerSolutions.P7

def sieveBytes (sieve : Array Bool) : List UInt8 :=
  sieve.toList.map fun flag => if flag then 1 else 0

/-- The Python buffer initialization encodes the existing Boolean sieve. -/
theorem initialSieve_bytes (limit : Nat) :
    sieveBytes (initialSieve limit) = (List.replicate (sieveSize limit) 1).set 0 0 := by
  apply List.ext_getElem
  · simp [sieveBytes, initialSieve]
  · intro index left right
    have valid : index < sieveSize limit := by simpa [sieveBytes, initialSieve] using left
    by_cases zero : index = 0
    · subst index
      simp [sieveBytes, initialSieve, List.getElem_set_self]
    · simp [sieveBytes, initialSieve, List.getElem_set_ne (Ne.symm zero), zero]

/-- The slice's arithmetic progression is exactly the odd multiples marked by the Lean sieve. -/
theorem mark_progression (i index : Nat) :
    (oddAt i * oddAt i / 2 ≤ index ∧ oddAt i ∣ index - oddAt i * oddAt i / 2) ↔
      oddAt i * oddAt i ≤ oddAt index ∧ oddAt index % oddAt i = 0 := by
  let p := oddAt i
  let start := p * p / 2
  have square : p * p = 2 * start + 1 := by
    have parity : p * p % 2 = 1 := by simp [p, oddAt, Nat.add_mod, Nat.mul_mod]
    have division := Nat.mod_add_div (p * p) 2
    omega
  have coprime : Nat.Coprime p 2 := Nat.coprime_two_right.mpr ⟨i, by simp [p, oddAt]⟩
  have squareMultiple : p ∣ p * p := dvd_mul_right p p
  change (start ≤ index ∧ p ∣ index - start) ↔ p * p ≤ oddAt index ∧ oddAt index % p = 0
  have lower : start ≤ index ↔ p * p ≤ oddAt index := by simp only [oddAt]; omega
  constructor
  · rintro ⟨bound, selected⟩
    have decomposition : oddAt index = p * p + 2 * (index - start) := by simp only [oddAt]; omega
    exact ⟨lower.mp bound, Nat.mod_eq_zero_of_dvd (by
      rw [decomposition, ← Nat.dvd_add_iff_right squareMultiple, coprime.dvd_mul_left]
      exact selected)⟩
  · rintro ⟨bound, marked⟩
    have bound := lower.mpr bound
    have decomposition : oddAt index = p * p + 2 * (index - start) := by simp only [oddAt]; omega
    have multiple := Nat.dvd_of_mod_eq_zero marked
    rw [decomposition, ← Nat.dvd_add_iff_right squareMultiple, coprime.dvd_mul_left] at multiple
    exact ⟨bound, multiple⟩

open FormalSW.Py.Builtin

private theorem length_setUniform (indices : List Nat) (values : List UInt8) (value : UInt8) :
    (indices.foldl (fun result index => result.set index value) values).length = values.length := by
  induction indices generalizing values with
  | nil => rfl
  | cons index indices ih => simpa only [List.foldl_cons, List.length_set] using ih (values.set index value)

/-- Clearing the byte slice implements the existing Boolean marking operation. -/
theorem markMultiples_bytes (i : Nat) (sieve : Array Bool) :
    let p := oddAt i
    let start := p * p / 2
    let positions := List.range' start ((sieve.size - start + p - 1) / p) p
    positions.foldl (fun bytes index => bytes.set index 0) (sieveBytes sieve) =
      sieveBytes (markMultiples p sieve) := by
  dsimp only
  let p := oddAt i
  let start := p * p / 2
  let positions := List.range' start ((sieve.size - start + p - 1) / p) p
  change positions.foldl (fun bytes index => bytes.set index 0) (sieveBytes sieve) =
    sieveBytes (markMultiples p sieve)
  have positive : 0 < p := by simp [p, oddAt]
  apply List.ext_getElem?
  intro index
  by_cases valid : index < sieve.size
  · rw [FormalSW.Py.Builtin.List.getElem?_setUniform positions (sieveBytes sieve) 0 index
      (by simpa [sieveBytes] using valid)]
    have selected : index ∈ positions ↔ p * p ≤ oddAt index ∧ oddAt index % p = 0 := by
      rw [Slice.mem_positive_end _ _ _ _ positive]
      constructor
      · rintro ⟨lower, _, multiple⟩
        exact (mark_progression i index).mp ⟨lower, multiple⟩
      · intro marked
        obtain ⟨lower, multiple⟩ := (mark_progression i index).mpr marked
        exact ⟨lower, valid, multiple⟩
    simp [sieveBytes, markMultiples, selected, valid, p]
    split_ifs <;> simp_all
    omega
  · have leftLength : (positions.foldl (fun bytes index => bytes.set index 0) (sieveBytes sieve)).length = sieve.size := by
      simp [length_setUniform, sieveBytes]
    have rightLength : (sieveBytes (markMultiples p sieve)).length = sieve.size := by
      simp [sieveBytes, markMultiples]
    rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by omega)]

open FormalSW.PyAst
open FormalSW.Py.Builtin

/-- The heap operation used by the source's extended slice marks exactly the Lean multiples. -/
theorem markSlice_buffer (heap : Mutable.Heap) (address i : Nat) (sieve : Array Bool)
    (marker : 1 ≤ i) (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    let p := oddAt i
    let start := p * p / 2
    let count := ((sieve.size : Int) - 1 - start) / p + 1
    Mutable.setSlice (.ref address) (.scalar (.int start)) (.scalar .none) (.scalar (.int p))
      (.scalar (.bytes (List.replicate count.toNat 0))) heap =
        some (.ok (.scalar .none), heap.set address (.bytearray (sieveBytes (markMultiples p sieve)))) := by
  dsimp only
  let p := oddAt i
  let start := p * p / 2
  have positive : 0 < p := by simp [p, oddAt]
  have nonzero : (p : Int) ≠ 0 := by omega
  have positiveInt : (0 : Int) < p := by omega
  have extended : (p : Int) ≠ 1 := by dsimp [p, oddAt]; omega
  rw [Slice.positive_end_count _ _ _ positive]
  apply Mutable.setSlice_buffer heap address (sieveBytes sieve) _ _ _ _ _
    { start := some start, step := some p } found
  · change (if (p : Int) = 0 then some (.error ⟨"ValueError", []⟩, heap)
      else some (.ok { start := some (start : Int), stop := none, step := some (p : Int) }, heap) :
        Option (Except ExceptionObject Slice.Slice × Mutable.Heap)) = _
    rw [if_neg nonzero]
  · have normalized : Slice.indices { start := some (start : Int), step := some (p : Int) }
        (sieveBytes sieve).length = .ok
          (max 0 (min (start : Int) (sieveBytes sieve).length), (sieveBytes sieve).length, (p : Int)) := by
      simp only [Slice.indices, Option.getD_some, nonzero, ↓reduceIte, positiveInt]
      change Except.ok
        (max 0 (min (if (start : Int) < 0 then (start : Int) + (sieveBytes sieve).length else start)
          (sieveBytes sieve).length), ((sieveBytes sieve).length : Int), (p : Int)) = _
      rw [if_neg (by omega : ¬ (start : Int) < 0)]
    have selected := Slice.indexList_positive_end (sieveBytes sieve).length start p positive
    have update := ByteArray.setSlice_uniform (sieveBytes sieve)
      { start := some start, step := some p } 0 normalized extended selected
    rw [← markMultiples_bytes i sieve]
    simpa only [sieveBytes, List.length_map, Array.length_toList, List.length_range'] using update

end ProjectEulerSolutions.P7

namespace solvers.«7»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerSolutions.P7
private abbrev program := Solvers.P7Ast.module

private theorem bits (n : Nat) (nonzero : n ≠ 0) :
    Math.bitLengthCall (.int n) = .ok (.int ((Nat.log2 n + 1 : Nat) : Int)) := by
  simp [Math.bitLengthCall, FormalSW.Py.Builtin.Int.bitLength, nonzero]

/-- The overflow-free source bound agrees with the existing integer Lean model. -/
theorem initial_limit.result (n : Nat) :
    ReturnsValue program "initial_limit" [.int n] (.int (ProjectEulerSolutions.P7.initialLimit n)) := by
  apply BlockReturns.function rfl rfl
  by_cases small : n < 6
  · apply BlockReturns.branch (value := .bool true)
    · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp, small])
    · simp only [ProjectEulerSolutions.P7.initialLimit, if_pos small]
      exact BlockReturns.return_value Evaluates.constant
  · apply BlockReturns.branch (value := .bool false)
    · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp, small])
    · apply BlockReturns.assign (value := .int (Nat.log2 n))
      · apply Evaluates.binary (av := .int ((Nat.log2 n + 1 : Nat) : Int)) (bv := .int 1)
        · apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
          exact congrArg some (bits n (by omega))
        · exact Evaluates.constant
        · simp [scalarBinary]
      · apply BlockReturns.return_value
        apply Evaluates.binary (av := .int ((n * (Nat.log2 n + (Nat.log2 (Nat.log2 n + 1) + 1)) : Nat) : Int))
          (bv := .int 10)
        · apply Evaluates.binary (av := .int n)
            (bv := .int ((Nat.log2 n + (Nat.log2 (Nat.log2 n + 1) + 1) : Nat) : Int))
          · exact Evaluates.lookup (by simp [ScalarEnv.set, argumentEnv])
          · apply Evaluates.binary (av := .int (Nat.log2 n))
              (bv := .int ((Nat.log2 (Nat.log2 n + 1) + 1 : Nat) : Int))
            · exact Evaluates.lookup (by simp [ScalarEnv.set])
            · apply Evaluates.builtin (values := [.int ((Nat.log2 n + 1 : Nat) : Int)]) (.cons
                (Evaluates.binary (av := .int (Nat.log2 n)) (bv := .int 1) (Evaluates.lookup (by simp [ScalarEnv.set])) Evaluates.constant
                  (by simp [scalarBinary])) .nil)
              exact congrArg some (bits (Nat.log2 n + 1) (by omega))
            · simp [scalarBinary]
          · simp [scalarBinary]
        · exact Evaluates.constant
        · simp [scalarBinary, ProjectEulerSolutions.P7.initialLimit, small, Nat.add_assoc]

theorem initial_limit.«Returns» (n : Int) : (function program "initial_limit" [.int n]).Returns := by
  by_cases nonnegative : 0 ≤ n
  · simpa [Int.toNat_of_nonneg nonnegative] using (initial_limit.result n.toNat).returns
  · apply ReturnsValue.returns (value := .int 15)
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (value := .bool true)
    · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp]; omega)
    · exact BlockReturns.return_value Evaluates.constant

/-- The retry body is extracted from the source-bound AST. -/
private def sieveBody : List Stmt :=
  match findFunctionIn program.body "solve" with
  | some (_, statements) => match statements.getLast? with
    | some (.whileDo _ body _) => body
    | _ => []
  | none => []

/-- The source allocates and initializes precisely the previously verified sieve. -/
theorem sieve.initialize (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (bound : env "limit" = some (.scalar (.int limit))) :
    HeapCalls.BlockContinues program env (sieveBody.take 3) heap
      (HeapCalls.setLocal (HeapCalls.setLocal env "size" (.scalar (.int (sieveSize limit))))
        "is_prime" (.ref (heap.length + 1)))
      (heap ++ [.bytearray [1]] ++ [.bytearray (sieveBytes (initialSieve limit))]) := by
  rw [initialSieve_bytes]
  apply HeapCalls.BlockContinues.append (body := [sieveBody[0]'(by decide)])
      (rest := [sieveBody[1]'(by decide), sieveBody[2]'(by decide)])
  · apply HeapCalls.BlockContinues.assign (value := .scalar (.int (sieveSize limit))) (finalHeap := heap)
    apply HeapCalls.Evaluates.binary (av := .scalar (.int (limit / 2 : Nat))) (bv := .scalar (.int 1)) (middle := heap) (afterRight := heap)
    · apply HeapCalls.Evaluates.binary (av := .scalar (.int limit)) (bv := .scalar (.int 2)) (middle := heap) (afterRight := heap)
      · exact HeapCalls.Evaluates.lookup bound
      · exact HeapCalls.Evaluates.constant
      · simp [Mutable.binary, scalarBinary]
    · exact HeapCalls.Evaluates.constant
    · simp [Mutable.binary, scalarBinary, sieveSize]
  · apply HeapCalls.BlockContinues.append (body := [sieveBody[1]'(by decide)]) (rest := [sieveBody[2]'(by decide)])
    · apply HeapCalls.BlockContinues.assign (value := .ref (heap.length + 1))
        (finalHeap := heap ++ [.bytearray [1]] ++ [.bytearray (List.replicate (sieveSize limit) 1)])
      apply HeapCalls.Evaluates.binary
        (av := .ref heap.length) (middle := heap ++ [.bytearray [1]])
        (bv := .scalar (.int (sieveSize limit))) (afterRight := heap ++ [.bytearray [1]])
      · apply HeapCalls.Evaluates.builtin (.cons HeapCalls.Evaluates.constant .nil)
        rfl
      · exact HeapCalls.Evaluates.lookup rfl
      · simp [Mutable.binary, Mutable.allocateBuffer, Except.map, show (BinOp.mult == .mult) = true from rfl]
    · apply HeapCalls.BlockContinues.setItem
      · intro a b c impossible; cases impossible
      · exact HeapCalls.Evaluates.constant
      · exact .cons (HeapCalls.Evaluates.lookup rfl) (.cons HeapCalls.Evaluates.constant .nil)
      · have found : (heap ++ [.bytearray [1]] ++ [.bytearray (List.replicate (sieveSize limit) 1)])[heap.length + 1]? =
            some (Mutable.Object.bytearray (List.replicate (sieveSize limit) 1)) := by simp
        have write := Mutable.setItem_buffer _ (heap.length + 1) 0 (List.replicate (sieveSize limit) 1) 0 found
          (by simp [sieveSize])
        simpa using write

/-- The true branch of the source sieve iteration. -/
private def markBody : List Stmt :=
  match (sieveBody[5]? : Option Stmt) with
  | some (.forIn _ _ [.ifThen _ body _] _) => body
  | _ => []

private def markEnv (env : HeapCalls.Env) (i size : Nat) : HeapCalls.Env :=
  let p := oddAt i
  let start := p * p / 2
  let count := ((size : Int) - 1 - start) / p + 1
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal env
    "p" (.scalar (.int p))) "start" (.scalar (.int start))) "step" (.scalar (.int p)))
    "count" (.scalar (.int count))

/-- Every assignment in the source marking branch implements the verified slice update. -/
theorem sieve.mark (env : HeapCalls.Env) (heap : Mutable.Heap) (address i : Nat) (sieve : Array Bool)
    (marker : 1 ≤ i)
    (index : env "i" = some (.scalar (.int i)))
    (size : env "size" = some (.scalar (.int sieve.size)))
    (buffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    HeapCalls.BlockContinues program env markBody heap (markEnv env i sieve.size)
      (heap.set address (.bytearray (sieveBytes (markMultiples (oddAt i) sieve)))) := by
  let p := oddAt i
  let start := p * p / 2
  let count := ((sieve.size : Int) - 1 - start) / p + 1
  let e1 := HeapCalls.setLocal env "p" (.scalar (.int p))
  let e2 := HeapCalls.setLocal e1 "start" (.scalar (.int start))
  let e3 := HeapCalls.setLocal e2 "step" (.scalar (.int p))
  let e4 := HeapCalls.setLocal e3 "count" (.scalar (.int count))
  have positive : 0 < p := by simp [p, oddAt]
  have first : HeapCalls.BlockContinues program env [markBody[0]'(by decide)] heap e1 heap := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (av := .scalar (.int (2 * i : Nat)))
      (bv := .scalar (.int 1)) (middle := heap) (afterRight := heap)
    · apply HeapCalls.Evaluates.binary (av := .scalar (.int 2)) (bv := .scalar (.int i))
        (middle := heap) (afterRight := heap)
      · exact HeapCalls.Evaluates.constant
      · exact HeapCalls.Evaluates.lookup index
      · simp [Mutable.binary, scalarBinary]
    · exact HeapCalls.Evaluates.constant
    · simp [Mutable.binary, scalarBinary, p, oddAt]
  have second : HeapCalls.BlockContinues program e1 [markBody[1]'(by decide)] heap e2 heap := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (av := .scalar (.int (p * p : Nat)))
      (bv := .scalar (.int 2)) (middle := heap) (afterRight := heap)
    · exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl)
        (HeapCalls.Evaluates.lookup rfl) (by simp [Mutable.binary, scalarBinary])
    · exact HeapCalls.Evaluates.constant
    · simp [Mutable.binary, scalarBinary, start]
  have third : HeapCalls.BlockContinues program e2 [markBody[2]'(by decide)] heap e3 heap := by
    exact HeapCalls.BlockContinues.assign (HeapCalls.Evaluates.lookup
      (by simp [e2, e1, HeapCalls.setLocal]))
  have fourth : HeapCalls.BlockContinues program e3 [markBody[3]'(by decide)] heap e4 heap := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (av := .scalar (.int (((sieve.size : Int) - 1 - start) / p)))
      (bv := .scalar (.int 1)) (middle := heap) (afterRight := heap)
    · apply HeapCalls.Evaluates.binary (av := .scalar (.int ((sieve.size : Int) - 1 - start)))
        (bv := .scalar (.int p)) (middle := heap) (afterRight := heap)
      · apply HeapCalls.Evaluates.binary (av := .scalar (.int ((sieve.size : Int) - 1)))
          (bv := .scalar (.int start)) (middle := heap) (afterRight := heap)
        · exact HeapCalls.Evaluates.binary (av := .scalar (.int sieve.size)) (bv := .scalar (.int 1))
            (HeapCalls.Evaluates.lookup (by simp [e3, e2, e1, HeapCalls.setLocal, size]))
            HeapCalls.Evaluates.constant rfl
        · exact HeapCalls.Evaluates.lookup (by simp [e3, e2, HeapCalls.setLocal])
        · rfl
      · exact HeapCalls.Evaluates.lookup rfl
      · simp [Mutable.binary, scalarBinary, Nat.ne_of_gt positive, positive]
    · exact HeapCalls.Evaluates.constant
    · rfl
  have fifth : HeapCalls.BlockContinues program e4 [markBody[4]'(by decide)] heap e4
      (heap.set address (.bytearray (sieveBytes (markMultiples p sieve)))) := by
    apply HeapCalls.BlockContinues.setSlice
      (value := .scalar (.bytes (List.replicate count.toNat 0))) (afterRhs := heap) (afterTarget := heap)
      (receiver := .ref address) (lowerValue := .scalar (.int start))
      (upperValue := .scalar .none) (strideValue := .scalar (.int p))
    · exact HeapCalls.Evaluates.binary HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup rfl)
        (by simp [Mutable.binary, scalarBinary])
    · exact .cons (HeapCalls.Evaluates.lookup (by simp [e4, e3, e2, e1, HeapCalls.setLocal, buffer]))
        (.cons (HeapCalls.Evaluates.lookup (by simp [e4, e3, e2, HeapCalls.setLocal]))
          (.cons HeapCalls.Evaluates.constant (.cons (HeapCalls.Evaluates.lookup
            (by simp [e4, e3, HeapCalls.setLocal])) .nil)))
    · exact markSlice_buffer heap address i sieve marker found
  exact first.append (second.append (third.append (fourth.append fifth)))

private def sieveIteration : List Stmt :=
  match (sieveBody[5]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

/-- A buffer test reads the corresponding Boolean flag without changing the heap. -/
theorem sieve.read (env : HeapCalls.Env) (heap : Mutable.Heap) (address i : Nat) (sieve : Array Bool)
    (valid : i < sieve.size)
    (index : env "i" = some (.scalar (.int i)))
    (buffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    HeapCalls.Evaluates program env (.subscript (.name "is_prime") (.name "i")) heap
      (.scalar (.int (if sieve[i] then 1 else 0))) heap := by
  apply HeapCalls.Evaluates.subscript (value := .ref address) (position := .scalar (.int i))
    (middle := heap) (afterIndex := heap)
  · intro a b c impossible; cases impossible
  · exact HeapCalls.Evaluates.lookup buffer
  · exact HeapCalls.Evaluates.lookup index
  · have read := Mutable.getItem_buffer heap address i (sieveBytes sieve) found (by simpa [sieveBytes] using valid)
    cases flag : sieve[i] <;> simpa [sieveBytes, flag] using read

/-- One source iteration agrees with the mathematical sieve step. -/
theorem sieve.iteration (env : HeapCalls.Env) (heap : Mutable.Heap) (address i : Nat) (sieve : Array Bool)
    (marker : 1 ≤ i) (valid : i < sieve.size)
    (index : env "i" = some (.scalar (.int i)))
    (size : env "size" = some (.scalar (.int sieve.size)))
    (buffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    HeapCalls.BlockContinues program env sieveIteration heap
      (if sieve[i] then markEnv env i sieve.size else env)
      (if sieve[i] then heap.set address (.bytearray (sieveBytes (markMultiples (oddAt i) sieve))) else heap) := by
  apply HeapCalls.BlockContinues.branch (value := .scalar (.int (if sieve[i] then 1 else 0)))
    (middle := heap) (flag := sieve[i])
  · exact solvers.«7».sieve.read env heap address i sieve valid index buffer found
  · cases flag : sieve[i] <;> rfl
  · cases flag : sieve[i]
    · exact HeapCalls.BlockContinues.nil
    · exact solvers.«7».sieve.mark env heap address i sieve marker index size buffer found

/-- The marking loop preserves the inputs and the buffer reference needed afterwards. -/
private def SieveStable (before after : HeapCalls.Env) : Prop :=
  ∀ name ∈ ["n", "limit", "size", "is_prime", "math"], after name = before name

private theorem SieveStable.refl (env : HeapCalls.Env) : SieveStable env env := by
  intro name member; rfl

private theorem SieveStable.trans {first middle last : HeapCalls.Env}
    (a : SieveStable first middle) (b : SieveStable middle last) : SieveStable first last := by
  intro name member
  exact (b name member).trans (a name member)

private theorem SieveStable.index (env : HeapCalls.Env) (i : Nat) :
    SieveStable env (HeapCalls.setLocal env "i" (.scalar (.int i))) := by
  intro name member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;> rfl

private theorem SieveStable.mark (env : HeapCalls.Env) (i size : Nat) :
    SieveStable env (markEnv env i size) := by
  intro name member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;> rfl

private theorem heap_set_same (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨valid, value⟩ := List.getElem?_eq_some_iff.mp found
  rw [← value, List.set_getElem_self valid]

/-- The source range loop executes precisely the existing terminating Lean sieve loop. -/
theorem sieve.loop (limit maxI next : Nat) (sieve : Array Bool) :
    ∀ (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat),
    1 ≤ next → maxI < sieve.size →
    env "size" = some (.scalar (.int sieve.size)) → env "is_prime" = some (.ref address) →
    heap[address]? = some (.bytearray (sieveBytes sieve)) →
    ∃ finalEnv,
      HeapCalls.BlockContinues program env
        [.rangeStepLoop "i" next (maxI + 1) 1 sieveIteration []] heap finalEnv
        (heap.set address (.bytearray (sieveBytes (sieveLoop limit maxI next sieve)))) ∧
      SieveStable env finalEnv := by
  induction next, sieve using sieveLoop.induct maxI with
  | case1 next sieve finished =>
      intro env heap address positive bounded size buffer found
      refine ⟨env, ?_, SieveStable.refl env⟩
      rw [sieveLoop.eq_def, if_pos finished, heap_set_same heap address _ found]
      exact HeapCalls.BlockContinues.range_done (by simp [Calls.rangeContinues]; omega)
        HeapCalls.BlockContinues.nil
  | case2 next sieve ongoing live p ih =>
      intro env heap address positive bounded size buffer found
      have valid : next < sieve.size := by omega
      have liveAt : sieve[next] = true := by simpa only [getElem!_pos sieve next valid] using live
      let indexed := HeapCalls.setLocal env "i" (.scalar (.int next))
      let updated := markMultiples (oddAt next) sieve
      let after := markEnv indexed next sieve.size
      let changed := heap.set address (.bytearray (sieveBytes updated))
      have stable : SieveStable env after := (SieveStable.index env next).trans (SieveStable.mark indexed next sieve.size)
      have addressValid : address < heap.length := (List.getElem?_eq_some_iff.mp found).choose
      obtain ⟨finalEnv, rest, preserved⟩ := ih after changed address (by omega)
        (by simpa [updated, markMultiples_size] using bounded)
        (by simpa [updated, markMultiples_size] using (stable "size" (by simp)).trans size)
        ((stable "is_prime" (by simp)).trans buffer)
        (by simp [changed, updated, p, List.getElem?_set_self addressValid])
      refine ⟨finalEnv, ?_, stable.trans preserved⟩
      rw [sieveLoop.eq_def, if_neg ongoing, if_pos live]
      apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have iteration := solvers.«7».sieve.iteration indexed heap address next sieve positive valid rfl
        (by simpa [indexed, HeapCalls.setLocal] using size)
        (by simpa [indexed, HeapCalls.setLocal] using buffer) found
      simp only [liveAt, ↓reduceIte] at iteration
      have combined := iteration.append rest
      simpa only [changed, List.set_set, Int.natCast_add, Int.natCast_one] using combined
  | case3 next sieve ongoing dead ih =>
      intro env heap address positive bounded size buffer found
      have valid : next < sieve.size := by omega
      have deadAt : sieve[next] = false := by
        have flag : sieve[next] ≠ true := by simpa only [getElem!_pos sieve next valid] using dead
        exact Bool.eq_false_iff.mpr flag
      let indexed := HeapCalls.setLocal env "i" (.scalar (.int next))
      have stable := SieveStable.index env next
      obtain ⟨finalEnv, rest, preserved⟩ := ih indexed heap address (by omega) bounded
        (by simpa [indexed, HeapCalls.setLocal] using size)
        (by simpa [indexed, HeapCalls.setLocal] using buffer) found
      refine ⟨finalEnv, ?_, stable.trans preserved⟩
      rw [sieveLoop.eq_def, if_neg ongoing, if_neg dead]
      apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have iteration := solvers.«7».sieve.iteration indexed heap address next sieve positive valid rfl
        (by simpa [indexed, HeapCalls.setLocal] using size)
        (by simpa [indexed, HeapCalls.setLocal] using buffer) found
      simp only [deadAt] at iteration
      simpa only [Int.natCast_add, Int.natCast_one] using iteration.append rest

/-- The source's `range(1, max_i + 1)` drives the verified marking loop. -/
theorem sieve.markLoop (limit maxI : Nat) (sieve : Array Bool)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (bounded : maxI < sieve.size)
    (maximum : env "max_i" = some (.scalar (.int maxI)))
    (size : env "size" = some (.scalar (.int sieve.size)))
    (buffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    ∃ finalEnv,
      HeapCalls.BlockContinues program env [sieveBody[5]'(by decide)] heap finalEnv
        (heap.set address (.bytearray (sieveBytes (sieveLoop limit maxI 1 sieve)))) ∧
      SieveStable env finalEnv := by
  obtain ⟨finalEnv, loop, stable⟩ := solvers.«7».sieve.loop limit maxI 1 sieve env heap address
    (by decide) bounded size buffer found
  refine ⟨finalEnv, ?_, stable⟩
  apply HeapCalls.BlockContinues.for_range (values := [.scalar (.int 1), .scalar (.int (maxI + 1 : Nat))])
    (middle := heap) (start := 1) (stop := maxI + 1) (stride := 1)
  · exact .cons HeapCalls.Evaluates.constant (.cons (HeapCalls.Evaluates.binary
      (HeapCalls.Evaluates.lookup maximum) HeapCalls.Evaluates.constant
      (by simp [Mutable.binary, scalarBinary])) .nil)
  · rfl
  · exact loop

private def scanBody : List Stmt :=
  match (sieveBody[7]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

/-- When the mathematical scan finds a prime, the source returns that same number. -/
theorem scan.returns (k i count : Nat) (sieve : Array Bool) :
    ∀ (env : HeapCalls.Env) (heap : Mutable.Heap) (address prime : Nat) (following : List Stmt),
    env "n" = some (.scalar (.int (k + 1 : Nat))) →
    env "count_primes" = some (.scalar (.int count)) →
    env "is_prime" = some (.ref address) →
    heap[address]? = some (.bytearray (sieveBytes sieve)) →
    scanNthPrimeIndex k i count sieve = some prime →
    HeapCalls.BlockReturns program env
      (.rangeStepLoop "i" i sieve.size 1 scanBody [] :: following) heap (.scalar (.int prime)) heap := by
  induction i, count using scanNthPrimeIndex.induct k sieve with
  | case1 i count finished =>
      intro env heap address prime following target counter buffer found result
      simp [scanNthPrimeIndex.eq_def, finished] at result
  | case2 i ongoing live =>
      intro env heap address prime following target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = true := by simpa only [getElem!_pos sieve i valid] using live
      have answer : prime = oddAt i := by simpa [scanNthPrimeIndex.eq_def, ongoing, live] using result.symm
      subst prime
      apply HeapCalls.BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply HeapCalls.BlockReturns.branch (value := .scalar (.int 1)) (flag := true) (middle := heap)
      · simpa [flag] using solvers.«7».sieve.read
          (HeapCalls.setLocal env "i" (.scalar (.int i))) heap address i sieve valid rfl
          (by simpa [HeapCalls.setLocal] using buffer) found
      · rfl
      · apply HeapCalls.BlockReturns.augAssign (left := .scalar (.int k))
          (right := .scalar (.int 1)) (value := .scalar (.int (k + 1 : Nat)))
          (middle := heap) (afterOperation := heap)
        · simpa [HeapCalls.setLocal] using counter
        · exact HeapCalls.Evaluates.constant
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        · apply HeapCalls.BlockReturns.branch (value := .scalar (.bool true)) (flag := true) (middle := heap)
          · exact HeapCalls.Evaluates.compare (av := .scalar (.int (k + 1 : Nat)))
              (bv := .scalar (.int (k + 1 : Nat))) (HeapCalls.Evaluates.lookup rfl)
              (HeapCalls.Evaluates.lookup (by simpa [HeapCalls.setLocal] using target))
              (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
          · rfl
          · apply HeapCalls.BlockReturns.return_value
            exact HeapCalls.Evaluates.binary (av := .scalar (.int (2 * i : Nat))) (bv := .scalar (.int 1))
              (middle := heap) (afterRight := heap)
              (HeapCalls.Evaluates.binary (av := .scalar (.int 2)) (bv := .scalar (.int i))
                HeapCalls.Evaluates.constant (HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal]))
                (by simp [Mutable.binary, scalarBinary]))
              HeapCalls.Evaluates.constant (by simp [Mutable.binary, scalarBinary, oddAt])
  | case3 i count ongoing live mismatch ih =>
      intro env heap address prime following target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = true := by simpa only [getElem!_pos sieve i valid] using live
      have resultNext : scanNthPrimeIndex k (i + 1) (count + 1) sieve = some prime := by
        rwa [scanNthPrimeIndex.eq_def, dif_neg ongoing, if_pos live, if_neg mismatch] at result
      apply HeapCalls.BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply HeapCalls.BlockReturns.branch (value := .scalar (.int 1)) (flag := true) (middle := heap)
      · simpa [flag] using solvers.«7».sieve.read
          (HeapCalls.setLocal env "i" (.scalar (.int i))) heap address i sieve valid rfl
          (by simpa [HeapCalls.setLocal] using buffer) found
      · rfl
      · apply HeapCalls.BlockReturns.augAssign (left := .scalar (.int count))
          (right := .scalar (.int 1)) (value := .scalar (.int (count + 1 : Nat)))
          (middle := heap) (afterOperation := heap)
        · simpa [HeapCalls.setLocal] using counter
        · exact HeapCalls.Evaluates.constant
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        · apply HeapCalls.BlockReturns.branch (value := .scalar (.bool false)) (flag := false) (middle := heap)
          · exact HeapCalls.Evaluates.compare (av := .scalar (.int (count + 1 : Nat)))
              (bv := .scalar (.int (k + 1 : Nat))) (HeapCalls.Evaluates.lookup rfl)
              (HeapCalls.Evaluates.lookup (by simpa [HeapCalls.setLocal] using target))
              (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, mismatch])
          · rfl
          · simpa only [Int.natCast_add, Int.natCast_one] using
              ih _ heap address prime following (by simpa [HeapCalls.setLocal] using target) rfl
                (by simpa [HeapCalls.setLocal] using buffer) found resultNext
  | case4 i count ongoing dead ih =>
      intro env heap address prime following target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = false := by
        have negative : sieve[i] ≠ true := by simpa only [getElem!_pos sieve i valid] using dead
        exact Bool.eq_false_iff.mpr negative
      have resultNext : scanNthPrimeIndex k (i + 1) count sieve = some prime := by
        rwa [scanNthPrimeIndex.eq_def, dif_neg ongoing, if_neg dead] at result
      apply HeapCalls.BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply HeapCalls.BlockReturns.branch (value := .scalar (.int 0)) (flag := false) (middle := heap)
      · simpa [flag] using solvers.«7».sieve.read
          (HeapCalls.setLocal env "i" (.scalar (.int i))) heap address i sieve valid rfl
          (by simpa [HeapCalls.setLocal] using buffer) found
      · rfl
      · simpa only [Int.natCast_add, Int.natCast_one] using
          ih _ heap address prime following (by simpa [HeapCalls.setLocal] using target)
            (by simpa [HeapCalls.setLocal] using counter)
            (by simpa [HeapCalls.setLocal] using buffer) found resultNext

private def scanHitBody : List Stmt :=
  match (scanBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

private theorem SieveStable.counter (env : HeapCalls.Env) (count : Nat) :
    SieveStable env (HeapCalls.setLocal env "count_primes" (.scalar (.int count))) := by
  intro name member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl <;> rfl

/-- An exhausted source scan preserves the inputs needed to retry with a larger limit. -/
theorem scan.continues (k i count : Nat) (sieve : Array Bool) :
    ∀ (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat),
    env "n" = some (.scalar (.int (k + 1 : Nat))) →
    env "count_primes" = some (.scalar (.int count)) →
    env "is_prime" = some (.ref address) →
    heap[address]? = some (.bytearray (sieveBytes sieve)) →
    scanNthPrimeIndex k i count sieve = none →
    ∃ finalEnv,
      HeapCalls.BlockContinues program env [.rangeStepLoop "i" i sieve.size 1 scanBody []]
        heap finalEnv heap ∧ SieveStable env finalEnv := by
  induction i, count using scanNthPrimeIndex.induct k sieve with
  | case1 i count finished =>
      intro env heap address target counter buffer found result
      refine ⟨env, ?_, SieveStable.refl env⟩
      exact HeapCalls.BlockContinues.range_done (by simp [Calls.rangeContinues]; omega)
        HeapCalls.BlockContinues.nil
  | case2 i ongoing live =>
      intro env heap address target counter buffer found result
      simp [scanNthPrimeIndex.eq_def, ongoing, live] at result
  | case3 i count ongoing live mismatch ih =>
      intro env heap address target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = true := by simpa only [getElem!_pos sieve i valid] using live
      have resultNext : scanNthPrimeIndex k (i + 1) (count + 1) sieve = none := by
        rwa [scanNthPrimeIndex.eq_def, dif_neg ongoing, if_pos live, if_neg mismatch] at result
      let indexed := HeapCalls.setLocal env "i" (.scalar (.int i))
      let advanced := HeapCalls.setLocal indexed "count_primes" (.scalar (.int (count + 1 : Nat)))
      have stable : SieveStable env advanced := (SieveStable.index env i).trans
        (SieveStable.counter indexed (count + 1))
      obtain ⟨finalEnv, rest, preserved⟩ := ih advanced heap address
        ((stable "n" (by simp)).trans target) rfl ((stable "is_prime" (by simp)).trans buffer) found resultNext
      refine ⟨finalEnv, ?_, stable.trans preserved⟩
      apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have advance : HeapCalls.BlockContinues program indexed (scanHitBody.take 1) heap advanced heap := by
        exact HeapCalls.BlockContinues.augAssign (left := .scalar (.int count))
          (by simpa [indexed, HeapCalls.setLocal] using counter) HeapCalls.Evaluates.constant
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
      have skip : HeapCalls.BlockContinues program advanced (scanHitBody.drop 1) heap advanced heap := by
        apply HeapCalls.BlockContinues.branch (value := .scalar (.bool false)) (flag := false) (middle := heap)
        · exact HeapCalls.Evaluates.compare (av := .scalar (.int (count + 1 : Nat)))
            (bv := .scalar (.int (k + 1 : Nat))) (HeapCalls.Evaluates.lookup rfl)
            (HeapCalls.Evaluates.lookup ((stable "n" (by simp)).trans target))
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, mismatch])
        · rfl
        · exact HeapCalls.BlockContinues.nil
      have iteration : HeapCalls.BlockContinues program indexed scanBody heap advanced heap := by
        apply HeapCalls.BlockContinues.branch (value := .scalar (.int 1)) (flag := true) (middle := heap)
        · simpa [flag] using solvers.«7».sieve.read indexed heap address i sieve valid rfl
            (by simpa [indexed, HeapCalls.setLocal] using buffer) found
        · rfl
        · exact advance.append skip
      simpa only [Int.natCast_add, Int.natCast_one] using iteration.append rest
  | case4 i count ongoing dead ih =>
      intro env heap address target counter buffer found result
      have valid : i < sieve.size := by omega
      have flag : sieve[i] = false := by
        have negative : sieve[i] ≠ true := by simpa only [getElem!_pos sieve i valid] using dead
        exact Bool.eq_false_iff.mpr negative
      have resultNext : scanNthPrimeIndex k (i + 1) count sieve = none := by
        rwa [scanNthPrimeIndex.eq_def, dif_neg ongoing, if_neg dead] at result
      let indexed := HeapCalls.setLocal env "i" (.scalar (.int i))
      have stable := SieveStable.index env i
      obtain ⟨finalEnv, rest, preserved⟩ := ih indexed heap address
        ((stable "n" (by simp)).trans target) (by simpa [indexed, HeapCalls.setLocal] using counter)
        ((stable "is_prime" (by simp)).trans buffer) found resultNext
      refine ⟨finalEnv, ?_, stable.trans preserved⟩
      apply HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      have iteration : HeapCalls.BlockContinues program indexed scanBody heap indexed heap := by
        apply HeapCalls.BlockContinues.branch (value := .scalar (.int 0)) (flag := false) (middle := heap)
        · simpa [flag] using solvers.«7».sieve.read indexed heap address i sieve valid rfl
            (by simpa [indexed, HeapCalls.setLocal] using buffer) found
        · rfl
        · exact HeapCalls.BlockContinues.nil
      simpa only [Int.natCast_add, Int.natCast_one] using iteration.append rest

/-- The source counter initialization and range scan return the mathematical result. -/
theorem scan.source_returns (k prime : Nat) (sieve : Array Bool) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (following : List Stmt)
    (target : env "n" = some (.scalar (.int (k + 1 : Nat))))
    (size : env "size" = some (.scalar (.int sieve.size)))
    (buffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve)))
    (result : scanNthPrimeIndex k 1 1 sieve = some prime) :
    HeapCalls.BlockReturns program env ((sieveBody.drop 6).take 2 ++ following) heap
      (.scalar (.int prime)) heap := by
  apply HeapCalls.BlockReturns.assign (value := .scalar (.int 1)) (middle := heap)
  · exact HeapCalls.Evaluates.constant
  · apply HeapCalls.BlockReturns.for_range (values := [.scalar (.int 1), .scalar (.int sieve.size)])
      (middle := heap) (start := 1) (stop := sieve.size) (stride := 1)
    · exact .cons HeapCalls.Evaluates.constant (.cons (HeapCalls.Evaluates.lookup
        (by simpa [HeapCalls.setLocal] using size)) .nil)
    · rfl
    · exact scan.returns k 1 1 sieve _ heap address prime following
        (by simpa [HeapCalls.setLocal] using target) rfl
        (by simpa [HeapCalls.setLocal] using buffer) found result

/-- An exhausted source scan falls through to the retry with its inputs intact. -/
theorem scan.source_continues (k : Nat) (sieve : Array Bool) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat)
    (target : env "n" = some (.scalar (.int (k + 1 : Nat))))
    (size : env "size" = some (.scalar (.int sieve.size)))
    (buffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve)))
    (result : scanNthPrimeIndex k 1 1 sieve = none) :
    ∃ finalEnv,
      HeapCalls.BlockContinues program env ((sieveBody.drop 6).take 2) heap finalEnv heap ∧
      SieveStable env finalEnv := by
  let initialized := HeapCalls.setLocal env "count_primes" (.scalar (.int 1))
  have stable := SieveStable.counter env 1
  obtain ⟨finalEnv, run, preserved⟩ := scan.continues k 1 1 sieve initialized heap address
    ((stable "n" (by simp)).trans target) rfl ((stable "is_prime" (by simp)).trans buffer) found result
  refine ⟨finalEnv, ?_, stable.trans preserved⟩
  apply HeapCalls.BlockContinues.append (body := [sieveBody[6]'(by decide)])
    (rest := [sieveBody[7]'(by decide)]) (middleEnv := initialized) (middleHeap := heap)
  · exact HeapCalls.BlockContinues.assign HeapCalls.Evaluates.constant
  · apply HeapCalls.BlockContinues.for_range (values := [.scalar (.int 1), .scalar (.int sieve.size)])
      (middle := heap) (start := 1) (stop := sieve.size) (stride := 1)
    · exact .cons HeapCalls.Evaluates.constant (.cons (HeapCalls.Evaluates.lookup
        ((stable "size" (by simp)).trans size)) .nil)
    · rfl
    · exact run

/-- The source computes its loop bound from the exact integer square root. -/
theorem sieve.rootSetup (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (bound : env "limit" = some (.scalar (.int limit)))
    (mathUnbound : env "math" = none) :
    HeapCalls.BlockContinues program env ((sieveBody.drop 3).take 2) heap
      (HeapCalls.setLocal (HeapCalls.setLocal env "root" (.scalar (.int (Nat.sqrt limit))))
        "max_i" (.scalar (.int (Nat.sqrt limit / 2 : Nat)))) heap := by
  have root : Math.isqrt limit = Nat.sqrt limit := Nat.eq_sqrt.mpr (Math.isqrt_spec limit)
  apply HeapCalls.BlockContinues.append (body := [sieveBody[3]'(by decide)])
    (rest := [sieveBody[4]'(by decide)]) (middleHeap := heap)
    (middleEnv := HeapCalls.setLocal env "root" (.scalar (.int (Nat.sqrt limit))))
  · apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.builtin (values := [.scalar (.int (Nat.sqrt limit))]) (middle := heap)
    · apply HeapCalls.Evaluates.Arguments.cons
      · apply HeapCalls.Evaluates.qualified_call (unbound := mathUnbound) (by decide) (.cons (HeapCalls.Evaluates.lookup bound) .nil)
        intro anchors stack
        apply HeapCalls.Reaches.next (following := ⟨.value (.scalar (.int (Nat.sqrt limit))), stack, heap⟩)
        · change HeapCalls.result anchors heap
            (HeapCalls.scalarResult (some (Math.isqrtCall [.int limit]))) stack = _
          simp [HeapCalls.result, HeapCalls.run, HeapCalls.scalarResult, Math.isqrtCall, root,
            show ¬ (limit : Int) < 0 by omega]
        · exact .refl
      · exact .nil
    · rfl
  · apply HeapCalls.BlockContinues.assign
    exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
      (by simp [Mutable.binary, scalarBinary])

private def sieveHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1]] ++ [.bytearray (sieveBytes (oddsOnlySieveImpl limit))]

/-- One source attempt constructs the same sieve as the existing Lean implementation. -/
theorem sieve.prepare (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (bound : env "limit" = some (.scalar (.int limit)))
    (mathUnbound : env "math" = none) :
    ∃ finalEnv,
      HeapCalls.BlockContinues program env (sieveBody.take 6) heap finalEnv (sieveHeap heap limit) ∧
      finalEnv "n" = env "n" ∧ finalEnv "limit" = env "limit" ∧
      finalEnv "size" = some (.scalar (.int (sieveSize limit))) ∧
      finalEnv "is_prime" = some (.ref (heap.length + 1)) ∧ finalEnv "math" = none := by
  let address := heap.length + 1
  let initialized := HeapCalls.setLocal (HeapCalls.setLocal env "size" (.scalar (.int (sieveSize limit))))
    "is_prime" (.ref address)
  let initialHeap := heap ++ [.bytearray [1]] ++ [.bytearray (sieveBytes (initialSieve limit))]
  let bounded := HeapCalls.setLocal (HeapCalls.setLocal initialized "root" (.scalar (.int (Nat.sqrt limit))))
    "max_i" (.scalar (.int (Nat.sqrt limit / 2 : Nat)))
  obtain ⟨finalEnv, marking, stable⟩ := solvers.«7».sieve.markLoop limit (Nat.sqrt limit / 2)
    (initialSieve limit) bounded initialHeap address
    (by simpa [initialSieve_size] using maxI_lt_sieveSize limit) rfl
    (by simp [bounded, initialized, HeapCalls.setLocal, initialSieve_size])
    (by simp [bounded, initialized, HeapCalls.setLocal])
    (by simp [initialHeap, address])
  refine ⟨finalEnv, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have initialization := solvers.«7».sieve.initialize limit env heap bound
    have root := solvers.«7».sieve.rootSetup limit initialized initialHeap
      (by simpa [initialized, HeapCalls.setLocal] using bound)
      (by simpa [initialized, HeapCalls.setLocal] using mathUnbound)
    have execution := initialization.append (root.append marking)
    simpa [initialHeap, address, sieveHeap, oddsOnlySieveImpl] using execution
  · simpa [bounded, initialized, HeapCalls.setLocal] using stable "n" (by simp)
  · simpa [bounded, initialized, HeapCalls.setLocal] using stable "limit" (by simp)
  · simpa [bounded, initialized, HeapCalls.setLocal] using stable "size" (by simp)
  · simpa [bounded, initialized, address, HeapCalls.setLocal] using stable "is_prime" (by simp)

  · have preserved : finalEnv "math" = env "math" := by
      simpa [bounded, initialized, HeapCalls.setLocal] using stable "math" (by simp)
    exact preserved.trans mathUnbound

/-- A successful source attempt returns the existing Lean model's candidate. -/
theorem sieve.attempt_returns (k limit prime : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (following : List Stmt)
    (target : env "n" = some (.scalar (.int (k + 2 : Nat))))
    (bound : env "limit" = some (.scalar (.int limit)))
    (mathUnbound : env "math" = none)
    (result : nthPrimeWithLimit (k + 1) limit = some prime) :
    HeapCalls.BlockReturns program env (sieveBody ++ following) heap (.scalar (.int prime))
      (sieveHeap heap limit) := by
  obtain ⟨ready, prepare, targetReady, limitReady, size, buffer, mathReady⟩ :=
    solvers.«7».sieve.prepare limit env heap bound mathUnbound
  have found : (sieveHeap heap limit)[heap.length + 1]? =
      some (Mutable.Object.bytearray (sieveBytes (oddsOnlySieveImpl limit))) := by simp [sieveHeap]
  have scanning := scan.source_returns (k + 1) prime (oddsOnlySieveImpl limit) ready
    (sieveHeap heap limit) (heap.length + 1) (sieveBody[8]'(by decide) :: following)
    (targetReady.trans target) (by simpa [oddsOnlySieveImpl_size] using size) buffer found result
  exact prepare.returns scanning

/-- An unsuccessful source attempt doubles its bound and preserves the requested index. -/
theorem sieve.attempt_continues (k limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (target : env "n" = some (.scalar (.int (k + 2 : Nat))))
    (bound : env "limit" = some (.scalar (.int limit)))
    (mathUnbound : env "math" = none)
    (result : nthPrimeWithLimit (k + 1) limit = none) :
    ∃ finalEnv,
      HeapCalls.BlockContinues program env sieveBody heap finalEnv (sieveHeap heap limit) ∧
      finalEnv "n" = some (.scalar (.int (k + 2 : Nat))) ∧
      finalEnv "limit" = some (.scalar (.int (limit * 2 : Nat))) ∧ finalEnv "math" = none := by
  obtain ⟨ready, prepare, targetReady, limitReady, size, buffer, mathReady⟩ :=
    solvers.«7».sieve.prepare limit env heap bound mathUnbound
  have found : (sieveHeap heap limit)[heap.length + 1]? =
      some (Mutable.Object.bytearray (sieveBytes (oddsOnlySieveImpl limit))) := by simp [sieveHeap]
  obtain ⟨scanned, scanRun, stable⟩ := scan.source_continues (k + 1) (oddsOnlySieveImpl limit) ready
    (sieveHeap heap limit) (heap.length + 1)
    (targetReady.trans target) (by simpa [oddsOnlySieveImpl_size] using size) buffer found result
  let finalEnv := HeapCalls.setLocal scanned "limit" (.scalar (.int (limit * 2 : Nat)))
  have grow : HeapCalls.BlockContinues program scanned [sieveBody[8]'(by decide)]
      (sieveHeap heap limit) finalEnv (sieveHeap heap limit) := by
    exact HeapCalls.BlockContinues.augAssign (left := .scalar (.int limit))
      ((stable "limit" (by simp)).trans (limitReady.trans bound)) HeapCalls.Evaluates.constant
      (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
  refine ⟨finalEnv, prepare.append (scanRun.append grow), ?_, rfl, ?_⟩
  · simpa [finalEnv, HeapCalls.setLocal] using (stable "n" (by simp)).trans (targetReady.trans target)
  · simpa [finalEnv, HeapCalls.setLocal] using (stable "math" (by simp)).trans mathReady

/-- Bertrand's bound supplies finite fuel for the source's otherwise unbounded retry loop. -/
theorem sieve.retry (k fuel limit : Nat)
    (enough : Nat.nth Nat.Prime (k + 1) ≤ limit * 2 ^ fuel) :
    ∀ (env : HeapCalls.Env) (heap : Mutable.Heap),
    env "n" = some (.scalar (.int (k + 2 : Nat))) →
    env "limit" = some (.scalar (.int limit)) →
    env "math" = none →
    ∃ finalHeap,
      HeapCalls.BlockReturns program env [.whileDo (.constant (.bool true)) sieveBody []] heap
        (.scalar (.int (Nat.nth Nat.Prime (k + 1)))) finalHeap := by
  induction fuel generalizing limit with
  | zero =>
      intro env heap target bound mathUnbound
      have result := nthPrimeWithLimit_eq_nth (k + 1) limit (by simpa using enough)
      refine ⟨sieveHeap heap limit, ?_⟩
      apply HeapCalls.BlockReturns.while_loop (value := .scalar (.bool true)) (flag := true) (middle := heap)
      · exact HeapCalls.Evaluates.constant
      · rfl
      · exact solvers.«7».sieve.attempt_returns k limit _ env heap _ target bound mathUnbound result
  | succ fuel ih =>
      intro env heap target bound mathUnbound
      cases result : nthPrimeWithLimit (k + 1) limit with
      | some prime =>
          have correct := nthPrimeWithLimit_sound result
          subst prime
          refine ⟨sieveHeap heap limit, ?_⟩
          apply HeapCalls.BlockReturns.while_loop (value := .scalar (.bool true)) (flag := true) (middle := heap)
          · exact HeapCalls.Evaluates.constant
          · rfl
          · exact solvers.«7».sieve.attempt_returns k limit _ env heap _ target bound mathUnbound result
      | none =>
          obtain ⟨advanced, attempt, targetNext, boundNext, mathNext⟩ :=
            solvers.«7».sieve.attempt_continues k limit env heap target bound mathUnbound result
          have enoughNext : Nat.nth Nat.Prime (k + 1) ≤ (limit * 2) * 2 ^ fuel := by
            simpa [pow_succ, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using enough
          obtain ⟨finalHeap, rest⟩ := ih (limit * 2) enoughNext advanced (sieveHeap heap limit) targetNext boundNext mathNext
          refine ⟨finalHeap, ?_⟩
          apply HeapCalls.BlockReturns.while_loop (value := .scalar (.bool true)) (flag := true) (middle := heap)
          · exact HeapCalls.Evaluates.constant
          · rfl
          · exact attempt.returns rest

private theorem heap_bits (n : Nat) (heap : Mutable.Heap) (nonzero : n ≠ 0) :
    HeapCalls.primitive (.builtin .intBitLength) [.scalar (.int n)] heap =
      some (.ok (.scalar (.int ((Nat.log2 n + 1 : Nat) : Int))), heap) := by
  change HeapCalls.scalarResult (some (Math.bitLengthCall (.int n))) heap = _
  rw [bits n nonzero]
  rfl

/-- The bound helper also preserves an arbitrary caller heap. -/
theorem initial_limit.heap_result (n : Nat) (heap : Mutable.Heap) :
    HeapCalls.ReturnsValue program "initial_limit" [.scalar (.int n)] heap
      (.scalar (.int (initialLimit n))) heap := by
  apply HeapCalls.BlockReturns.function rfl rfl
  by_cases small : n < 6
  · apply HeapCalls.BlockReturns.branch (value := .scalar (.bool true)) (flag := true) (middle := heap)
    · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])
    · rfl
    · simp only [initialLimit, if_pos small]
      exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  · apply HeapCalls.BlockReturns.branch (value := .scalar (.bool false)) (flag := false) (middle := heap)
    · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])
    · rfl
    · apply HeapCalls.BlockReturns.assign (value := .scalar (.int (Nat.log2 n))) (middle := heap)
      · apply HeapCalls.Evaluates.binary (av := .scalar (.int ((Nat.log2 n + 1 : Nat) : Int)))
          (bv := .scalar (.int 1)) (middle := heap) (afterRight := heap)
        · exact HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup rfl) .nil)
            (heap_bits n heap (by omega))
        · exact HeapCalls.Evaluates.constant
        · simp [Mutable.binary, scalarBinary]
      · apply HeapCalls.BlockReturns.return_value
        apply HeapCalls.Evaluates.binary
          (av := .scalar (.int ((n * (Nat.log2 n + (Nat.log2 (Nat.log2 n + 1) + 1)) : Nat) : Int)))
          (bv := .scalar (.int 10)) (middle := heap) (afterRight := heap)
        · apply HeapCalls.Evaluates.binary (av := .scalar (.int n))
            (bv := .scalar (.int ((Nat.log2 n + (Nat.log2 (Nat.log2 n + 1) + 1) : Nat) : Int)))
            (middle := heap) (afterRight := heap)
          · exact HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal, HeapCalls.argumentEnv])
          · apply HeapCalls.Evaluates.binary (av := .scalar (.int (Nat.log2 n)))
              (bv := .scalar (.int ((Nat.log2 (Nat.log2 n + 1) + 1 : Nat) : Int)))
              (middle := heap) (afterRight := heap)
            · exact HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal])
            · apply HeapCalls.Evaluates.builtin (values := [.scalar (.int ((Nat.log2 n + 1 : Nat) : Int))]) (middle := heap)
                (.cons (HeapCalls.Evaluates.binary (av := .scalar (.int (Nat.log2 n)))
                  (bv := .scalar (.int 1)) (HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal]))
                  HeapCalls.Evaluates.constant (by simp [Mutable.binary, scalarBinary])) .nil)
              exact heap_bits (Nat.log2 n + 1) heap (by omega)
            · simp [Mutable.binary, scalarBinary]
          · simp [Mutable.binary, scalarBinary]
        · exact HeapCalls.Evaluates.constant
        · simp [Mutable.binary, scalarBinary, initialLimit, small, Nat.add_assoc]

/-- The source's early return handles the first prime before allocating the sieve. -/
theorem solve.one :
    HeapCalls.ReturnsValue program "solve" [.scalar (.int 1)] [] (.scalar (.int 2)) [] := by
  intro anchors stack
  repeat' first | exact .refl | apply HeapCalls.Reaches.next rfl

/-- The complete source execution returns the previously specified nth prime. -/
theorem solve.result (n : Nat) (positive : 1 ≤ n) :
    ∃ heap, HeapCalls.ReturnsValue program "solve" [.scalar (.int n)] []
      (.scalar (.int (ProjectEulerStatements.P7.naive n))) heap := by
  rcases n with _ | (_ | k)
  · omega
  · refine ⟨[], ?_⟩
    simpa [ProjectEulerStatements.P7.naive, Nat.nth_prime_zero_eq_two] using solve.one
  · have enough : Nat.nth Nat.Prime (k + 1) ≤ initialLimit (k + 2) * 2 ^ (k + 2) := by
      calc
        Nat.nth Nat.Prime (k + 1) ≤ 2 ^ (k + 2) := nth_prime_le_two_pow_succ (k + 1)
        _ ≤ initialLimit (k + 2) * 2 ^ (k + 2) := by
          simpa using Nat.mul_le_mul_right (2 ^ (k + 2)) (one_le_initialLimit (k + 2))
    let env := HeapCalls.argumentEnv [{name := "n", annotation := some (.name "int")}]
      [.scalar (.int (k + 2 : Nat))]
    let ready := HeapCalls.setLocal env "limit" (.scalar (.int (initialLimit (k + 2))))
    obtain ⟨heap, retry⟩ := solvers.«7».sieve.retry k (k + 2) (initialLimit (k + 2)) enough ready [] rfl rfl rfl
    refine ⟨heap, ?_⟩
    apply HeapCalls.BlockReturns.function rfl rfl
    apply HeapCalls.BlockReturns.assertion (value := .scalar (.bool true)) (middle := [])
    · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply HeapCalls.BlockReturns.branch (value := .scalar (.bool false)) (flag := false) (middle := [])
      · exact HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
      · rfl
      · apply HeapCalls.BlockReturns.assign (value := .scalar (.int (initialLimit (k + 2)))) (middle := [])
        · exact HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup rfl) .nil)
            (initial_limit.heap_result (k + 2) [])
        · exact retry

theorem solve.«Returns» (n : Int) (requires : Calls.Condition program
    (fun name => if name = "n" then some (.int n) else none) Solvers.P7Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int n]).Returns := by
  have positive : 1 ≤ n := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare requires rfl
  lift n to Nat using (by omega : 0 ≤ n)
  obtain ⟨heap, run⟩ := solve.result n (by exact_mod_cast positive)
  exact run.scalar_returns

/-- Both source executions return the same nth prime, reusing the existing Lean specification. -/
theorem solve.«Equiv(naive)» (n : Int) (requires : Calls.Condition program
    (fun name => if name = "n" then some (.int n) else none) Solvers.P7Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int n]).Equiv (Calls.function P7Ast.module "naive" [.int n]) := by
  have positive : 1 ≤ n := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare requires rfl
  lift n to Nat using (by omega : 0 ≤ n)
  obtain ⟨heap, optimized⟩ := solve.result n (by exact_mod_cast positive)
  have optimizedRun := HeapCalls.ReturnsValue.scalar_finishes (values := [.int n]) optimized
  have referenceRun := (data.«project-euler-statements».data.py.project_euler_statements.p7.naive.result n).finishes
  intro outcome
  constructor
  · intro run
    rw [run.unique optimizedRun]
    exact referenceRun
  · intro run
    rw [run.unique referenceRun]
    exact optimizedRun

theorem solve.«Supported» :
    (∀ n : Int, Calls.Condition program (fun name => if name = "n" then some (.int n) else none)
      Solvers.P7Ast.requires_solve_0 → (HeapCalls.scalarFunction program "solve" [.int n]).Supported) ∧
    (∀ n : Int, Calls.Condition program (fun name => if name = "n" then some (.int n) else none)
      Solvers.P7Ast.requires_solve_0 → (Calls.function P7Ast.module "naive" [.int n]).Supported) :=
  ⟨fun n requires => (solve.«Returns» n requires).supported,
   fun n requires => ((solve.«Equiv(naive)» n requires).returns_iff.mp (solve.«Returns» n requires)).supported⟩

end solvers.«7»
