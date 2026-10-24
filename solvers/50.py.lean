import «.formalsw».solvers.«50.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.LoopCalls
import ProjectEulerSolutions.Lib.PrimeSieve
import ProjectEulerSolutions.Lib.BooleanPrimeSieve
import ProjectEulerSolutions.Lib.ConsecutivePrimeSums
import ProjectEulerSolutions.Lib.HeapCollection
import data.«project-euler-statements».data.py.project_euler_statements.«p50.py»

set_option Elab.async false

namespace solvers.«50»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P50Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

namespace sieve_is_prime
open ProjectEulerSolutions.PrimeSieve
private def code : List Stmt := executableBody ((findFunctionIn program.body "sieve_is_prime").getD ([], [])).2
private def loopBody : List Stmt := match (code[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def markBody : List Stmt := match (loopBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body | _ => []
private def writeBody : List Stmt :=
  [.assign [.subscript (.name "is_prime") (.name "m")] (.constant (.int 0))]

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

private def clearRange (bytes : List UInt8) (first stop stride : Nat) : List UInt8 :=
  (List.range' first ((stop - first + stride - 1) / stride) stride).foldl (fun result i => result.set i 0) bytes

private theorem clearRange_step (bytes : List UInt8) (first stop stride : Nat)
    (positive : 0 < stride) (active : first < stop) :
    clearRange bytes first stop stride = clearRange (bytes.set first 0) (first + stride) stop stride := by
  simp only [clearRange, ProjectEulerSolutions.BooleanPrimeSieve.count_next first stop stride positive active,
    List.range'_succ, List.foldl_cons]

private theorem clearRange_done (bytes : List UInt8) (first stop stride : Nat)
    (positive : 0 < stride) (finished : stop ≤ first) : clearRange bytes first stop stride = bytes := by
  simp [clearRange, Nat.sub_eq_zero_of_le finished, Nat.div_eq_of_lt (show stride - 1 < stride by omega)]

/-- Each source iteration writes one byte, retaining the original buffer address. -/
private theorem clear_loop (first stop stride address : Nat) (positive : 0 < stride) (bytes : List UInt8)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (size : stop ≤ bytes.length)
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "m" first stop stride writeBody []]
      heap finalEnv (heap.set address (.bytearray (clearRange bytes first stop stride))) ∧
      ∀ name, name ≠ "m" → finalEnv name = env name := by
  induction measure : stop - first using Nat.strong_induction_on generalizing first bytes heap env with
  | h measure ih =>
      by_cases active : first < stop
      · let bound := setLocal env "m" (number first)
        let middle := heap.set address (.bytearray (bytes.set first 0))
        have boundArray : bound "is_prime" = some (.ref address) := by simpa [bound, setLocal] using getArray
        have written : BlockContinues program bound writeBody heap bound middle := by
          apply BlockContinues.setItem (value := number 0) (receiver := .ref address) (subscript := number first)
            (by intros; simp) Evaluates.constant
            (.cons (Evaluates.lookup boundArray) (.cons (Evaluates.lookup (by simp [bound, setLocal])) .nil))
          exact Mutable.setItem_buffer heap address first bytes 0 found (by omega)
        obtain ⟨finalEnv, rest, stable⟩ := ih (stop - (first + stride)) (by omega) (first + stride)
          (bytes.set first 0) middle bound (by simpa using size) boundArray
          (List.getElem?_set_self (heap_inside found)) rfl
        refine ⟨finalEnv, ?_, ?_⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_eq_true,
            if_false, Int.natCast_add, clearRange_step bytes first stop stride positive active,
            middle, List.set_set] using written.append rest
        · intro name different
          exact (stable name different).trans (by simp [bound, setLocal, different])
      · refine ⟨env, ?_, fun _ _ => rfl⟩
        rw [clearRange_done bytes first stop stride positive (by omega), heap_set_same heap address _ found]
        exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil

private def markEnv (env : HeapCalls.Env) (p : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "start" (number (p * p : Nat))) "step" (number p)

private theorem mark_result (limit p address : Nat) (bytes : List UInt8) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (lower : 2 ≤ p) (size : bytes.length = limit + 1)
    (getP : env "p" = some (number p)) (getLimit : env "limit" = some (number limit))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv, BlockContinues program env markBody heap finalEnv (heap.set address (.bytearray (mark bytes p))) ∧
      finalEnv "limit" = env "limit" ∧ finalEnv "is_prime" = env "is_prime" := by
  let firstEnv := setLocal env "start" (number (p * p : Nat))
  let ready := markEnv env p
  have first : BlockContinues program env [markBody[0]'(by decide)] heap firstEnv heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP)
      (by simp [Mutable.binary, scalarBinary])
  have second : BlockContinues program firstEnv [markBody[1]'(by decide)] heap ready heap :=
    BlockContinues.assign (Evaluates.lookup (by simpa [firstEnv, setLocal] using getP))
  have readyArray : ready "is_prime" = some (.ref address) := by simpa [ready, markEnv, setLocal] using getArray
  obtain ⟨finalEnv, scanned, stable⟩ := clear_loop (p * p) (limit + 1) p address (by omega) bytes heap ready
    (by omega) readyArray found
  have third : BlockContinues program ready [markBody[2]'(by decide)] heap finalEnv
      (heap.set address (.bytearray (mark bytes p))) := by
    apply BlockContinues.for_range (values := [number (p * p : Nat), number (limit + 1 : Nat), number p])
      (start := (p * p : Nat)) (stop := (limit + 1 : Nat)) (stride := p) (middle := heap)
      (.cons (Evaluates.lookup (by simp [ready, markEnv, setLocal]))
        (.cons (Evaluates.binary (av := number limit) (bv := number 1)
          (Evaluates.lookup (by simpa [ready, markEnv, setLocal] using getLimit)) Evaluates.constant
          (by simp [Mutable.binary, scalarBinary]))
          (.cons (Evaluates.lookup (by simp [ready, markEnv, setLocal])) .nil)))
      (by
        change (if (p : Int) = 0 then Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)
          else Except.ok (((p * p : Nat) : Int), ((limit + 1 : Nat) : Int), (p : Int))) = _
        rw [if_neg (show (p : Int) ≠ 0 by omega)])
    simpa only [mark, markedIndices, clearRange, size] using scanned
  refine ⟨finalEnv, first.append (second.append third), ?_, ?_⟩
  · exact (stable "limit" (by decide)).trans (by simp [ready, markEnv, setLocal])
  · exact (stable "is_prime" (by decide)).trans (by simp [ready, markEnv, setLocal])

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
  after "limit" = before "limit" ∧ after "is_prime" = before "is_prime"

private theorem read_buffer (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat)
    (bytes : List UInt8) (valid : p < bytes.length) (getP : env "p" = some (number p))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    Evaluates program env (.subscript (.name "is_prime") (.name "p")) heap (number bytes[p].toNat) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getArray) (Evaluates.lookup getP)
  exact Mutable.getItem_buffer heap address p bytes found valid

private theorem iteration (limit p : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (large : 1 ≤ limit) (lower : 2 ≤ p) (inside : p * p ≤ limit)
    (getP : env "p" = some (number p)) (getLimit : env "limit" = some (number limit))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes))
    (invariant : Invariant limit p bytes) :
    ∃ finalEnv finalBytes, BlockContinues program env loopBody heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Invariant limit (p + 1) finalBytes ∧ Stable env finalEnv := by
  have size := invariant.1.1
  have valid : p < bytes.length := by nlinarith
  have live : bytes[p].toNat ≠ 0 ↔ Nat.Prime p :=
    (invariant.1.2 p bytes[p] lower (List.getElem?_eq_getElem valid)).trans (survives_current p lower)
  by_cases prime : Nat.Prime p
  · obtain ⟨finalEnv, marked, savedLimit, savedArray⟩ := mark_result limit p address bytes heap env
      lower size getP getLimit getArray found
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
    (getLimit : env "limit" = some (number limit)) (getArray : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (invariant : Invariant limit first bytes) :
    ∃ finalEnv finalBytes, BlockContinues program env
      [.rangeStepLoop "p" first (Nat.sqrt limit + 1) 1 loopBody []] heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Invariant limit (Nat.sqrt limit + 1) finalBytes ∧ Stable env finalEnv := by
  induction count generalizing first bytes env heap with
  | zero =>
      have same : first = Nat.sqrt limit + 1 := by omega
      refine ⟨env, bytes, ?_, by simpa only [same] using invariant, rfl, rfl⟩
      rw [heap_set_same heap address _ found]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "p" (number first)
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

private theorem initial_updates (limit : Nat) (large : 1 ≤ limit) :
    ((List.replicate (limit + 1) (1 : UInt8)).set 0 0).set 1 0 = initial limit := by
  obtain ⟨k, rfl⟩ : ∃ k, limit = k + 1 := ⟨limit - 1, by omega⟩
  simp [initial, List.replicate_succ]

private def initialHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray (initial limit)]

private theorem setup (limit : Nat) (large : 1 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (number limit)) :
    BlockContinues program env ((code.drop 1).take 3) heap
      (setLocal env "is_prime" (.ref (heap.length + 1))) (initialHeap heap limit) := by
  let ready := setLocal env "is_prime" (.ref (heap.length + 1))
  let allocated := heap ++ [.bytearray [1], .bytearray (List.replicate (limit + 1) 1)]
  let zeroed := heap ++ [.bytearray [1], .bytearray ((List.replicate (limit + 1) 1).set 0 0)]
  have first : BlockContinues program env [code[1]'(by decide)] heap ready allocated := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := .ref heap.length) (bv := number (limit + 1 : Nat))
      (middle := heap ++ [.bytearray [1]]) (afterRight := heap ++ [.bytearray [1]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
    · simp [allocated, Mutable.binary, Mutable.allocateBuffer, Except.map, show (BinOp.mult == .mult) = true from rfl]
  have second : BlockContinues program ready [code[2]'(by decide)] allocated ready zeroed := by
    apply BlockContinues.setItem (value := number 0) (by intros; simp) Evaluates.constant
      (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant .nil))
    simpa [allocated, zeroed] using Mutable.setItem_buffer allocated (heap.length + 1) 0
      (List.replicate (limit + 1) 1) 0 (by simp [allocated]) (by simp)
  have third : BlockContinues program ready [code[3]'(by decide)] zeroed ready (initialHeap heap limit) := by
    apply BlockContinues.branch (flag := true) (middle := zeroed) (value := .scalar (.bool true))
    · exact Evaluates.compare (Evaluates.lookup (by simpa [ready, setLocal] using getLimit)) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply BlockContinues.setItem (value := number 0) (by intros; simp) Evaluates.constant
        (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant .nil))
      simpa [zeroed, initialHeap, initial_updates limit large] using Mutable.setItem_buffer zeroed (heap.length + 1) 1
        ((List.replicate (limit + 1) 1).set 0 0) 0 (by simp [zeroed]) (by simp; omega)
  exact first.append (second.append third)

/-- The sieve returns an inclusive primality table and preserves every incoming object. -/
theorem positive_result (limit : Nat) (large : 1 ≤ limit) (heap : Mutable.Heap) :
    ∃ bytes, ReturnsValue program "sieve_is_prime" [number limit] heap (.ref (heap.length + 1))
      (heap ++ [.bytearray [1], .bytearray bytes]) ∧ bytes.length = limit + 1 ∧
      (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  let env := functionEnv ((findFunctionIn program.body "sieve_is_prime").getD ([], [])).1
    ((findFunctionIn program.body "sieve_is_prime").getD ([], [])).2 [number limit] []
  let ready := setLocal env "is_prime" (.ref (heap.length + 1))
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
  have rootPositive : 1 ≤ Nat.sqrt limit := by simpa using Nat.sqrt_le_sqrt large
  obtain ⟨finalEnv, bytes, executed, sieved, stable⟩ := loop limit 2 (Nat.sqrt limit - 1) (initial limit)
    ready (initialHeap heap limit) (heap.length + 1) large (by omega) (by omega)
    (by simpa [ready, setLocal] using (show env "limit" = some (number limit) from rfl))
    (by simp [ready, setLocal]) (by simp [initialHeap]) initialInvariant
  have loopRun : BlockContinues program ready [code[4]'(by decide)] (initialHeap heap limit) finalEnv
      (heap ++ [.bytearray [1], .bytearray bytes]) := by
    apply BlockContinues.for_range (middle := initialHeap heap limit)
      (values := [number 2, number (Nat.sqrt limit + 1 : Nat)])
      (.cons Evaluates.constant (.cons (Evaluates.binary (av := number (Nat.sqrt limit)) (bv := number 1)
        (middle := initialHeap heap limit) (afterRight := initialHeap heap limit)
        (by
          apply Evaluates.library_call (.cons (Evaluates.lookup
            (show ready "limit" = some (number limit) from rfl)) .nil) rfl
          change scalarResult (some (Math.isqrtCall [.int limit])) (initialHeap heap limit) = _
          have root : Math.isqrt limit = Nat.sqrt limit := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
          simp [scalarResult, Math.isqrtCall, root, show ¬ (limit : Int) < 0 by omega])
        Evaluates.constant (by simp [Mutable.binary, scalarBinary])) .nil)) rfl
    simpa [initialHeap] using executed
  refine ⟨bytes, ?_, sieved.1.1, ?_⟩
  · apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := false) (middle := heap) (value := .scalar (.bool false))
    · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply BlockContinues.returns ((setup limit large env heap rfl).append loopRun)
      exact BlockReturns.return_value (Evaluates.lookup (stable.2.trans (by simp [ready, setLocal])))
  · intro i byte found
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

/-- Zero returns one zero byte; negative limits return an empty buffer. -/
theorem nonpositive_result (limit : Int) (small : limit < 1) (heap : Mutable.Heap) :
    ReturnsValue program "sieve_is_prime" [number limit] heap (.ref (heap.length + 1))
      (heap ++ [.bytearray [0], .bytearray (List.replicate (limit + 1).toNat 0)]) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := true) (middle := heap) (value := .scalar (.bool true))
  · exact Evaluates.compare (av := number limit) (bv := number 1) (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])
  · rfl
  · apply BlockReturns.return_value
    apply Evaluates.binary (av := .ref heap.length) (bv := number (limit + 1))
      (middle := heap ++ [.bytearray [0]]) (afterRight := heap ++ [.bytearray [0]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.binary (av := number limit) (bv := number 1) (Evaluates.lookup rfl) Evaluates.constant rfl
    · simp [Mutable.binary, Mutable.allocateBuffer, Except.map, show (BinOp.mult == .mult) = true from rfl]

/-- All signed inputs return a correctly sized table whose truth values are exactly primality. -/
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ seed bytes, ReturnsValue program "sieve_is_prime" [number limit] heap (.ref (heap.length + 1))
      (heap ++ [.bytearray seed, .bytearray bytes]) ∧ bytes.length = (limit + 1).toNat ∧
      (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  by_cases small : limit < 1
  · refine ⟨[0], List.replicate (limit + 1).toNat 0, nonpositive_result limit small heap, by simp, ?_⟩
    intro i byte found
    have inside := (List.getElem?_eq_some_iff.mp found).1
    simp only [List.length_replicate] at inside
    have same : i = 0 := by omega
    have zero : byte = 0 := List.eq_of_mem_replicate (List.mem_of_getElem? found)
    simp [same, zero, Nat.not_prime_zero]
  · lift limit to Nat using (by omega : 0 ≤ limit)
    obtain ⟨bytes, returned, size, correct⟩ := positive_result limit (by omega) heap
    exact ⟨[1], bytes, returned, by simpa using size, correct⟩

theorem «Returns» (limit : Int) : (HeapCalls.call program "sieve_is_prime" [number limit] []).Returns := by
  obtain ⟨seed, bytes, returned, size, correct⟩ := result limit []
  exact returned.returns

theorem «Supported» (limit : Int) : (HeapCalls.call program "sieve_is_prime" [number limit] []).Supported :=
  («Returns» limit).supported

end sieve_is_prime

namespace longest_consecutive_prime_sum
open ProjectEulerSolutions.HeapCollection
open ProjectEulerSolutions.ConsecutivePrimeSums
private abbrev Reference := data.«project-euler-statements».data.py.project_euler_statements.p50.primes_below.values
private def code : List Stmt := executableBody ((findFunctionIn program.body "longest_consecutive_prime_sum").getD ([], [])).2
private def numbers (items : List Nat) : List Mutable.Value := items.map (fun (n : Nat) => number n)
private def primeBody (collector : Nat) : List Stmt :=
  [.ifThen (.subscript (.name "is_prime") (.name "i")) [.comprehensionAppend collector (.name "i")] []]

def primes (limit : Nat) : List Nat := (List.range' 2 (limit - 2)).filter fun n => decide (Nat.Prime n)

theorem primes_reference (limit : Nat) (large : 2 ≤ limit) : primes limit = Reference limit := by
  have split : List.range limit = [0, 1] ++ List.range' 2 (limit - 2) := by
    calc
      List.range limit = List.range' 0 (2 + (limit - 2)) := by
        rw [List.range_eq_range']
        congr 1
        omega
      _ = List.range' 0 2 ++ List.range' 2 (limit - 2) := by
        simpa using (List.range'_append_1 (s := 0) (m := 2) (n := limit - 2)).symm
      _ = [0, 1] ++ List.range' 2 (limit - 2) := rfl
  change primes limit = (List.range limit).filter (fun n => decide (Nat.Prime n))
  simp [primes, split, Nat.not_prime_zero, Nat.not_prime_one]

theorem primes_ordered (limit : Nat) (large : 2 ≤ limit) : (primes limit).Pairwise (· ≤ ·) := by
  rw [primes_reference limit large]
  exact List.pairwise_le_range.filter _

theorem primes_member (limit value : Nat) (large : 2 ≤ limit) :
    value ∈ primes limit ↔ value < limit ∧ Nat.Prime value := by
  rw [primes_reference limit large]
  change value ∈ (List.range limit).filter (fun n => decide (Nat.Prime n)) ↔ _
  simp

private theorem read_prime (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat) (bytes : List UInt8)
    (inside : p < bytes.length) (getP : env "i" = some (number p))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    Evaluates program env (.subscript (.name "is_prime") (.name "i")) heap (number bytes[p].toNat) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getArray) (Evaluates.lookup getP)
  exact Mutable.getItem_buffer heap address p bytes found inside

private theorem prime_loop (first count stop : Nat) (length : first + count = stop)
    (base : Mutable.Heap) (address : Nat) (bytes : List UInt8) (size : bytes.length = stop)
    (found : base[address]? = some (.bytearray bytes))
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i))
    (items : List Nat) (env : HeapCalls.Env) (getArray : env "is_prime" = some (.ref address)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "i" first stop 1 (primeBody base.length) []]
      (base ++ [.list (numbers items)]) finalEnv
      (base ++ [.list (numbers (items ++ (List.range' first count).filter fun p => decide (Nat.Prime p)))]) := by
  induction count generalizing first items env with
  | zero =>
      refine ⟨env, ?_⟩
      simp only [List.range'_zero, List.filter_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "i" (number first)
      have getArray' : bound "is_prime" = some (.ref address) := by simpa [bound, setLocal] using getArray
      have inside : first < bytes.length := by omega
      have tested := read_prime (base ++ [.list (numbers items)]) bound address first bytes inside
        (by simp [bound, setLocal]) getArray' (prefix_found base _ address _ found)
      have live := correct first bytes[first] (List.getElem?_eq_getElem inside)
      by_cases prime : Nat.Prime first
      · have emitted : BlockContinues program bound (primeBody base.length)
            (base ++ [.list (numbers items)]) bound (base ++ [.list (numbers (items ++ [first]))]) := by
          apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, live.mpr prime])
          have looked : Evaluates program bound (.name "i") (working base (numbers items) [])
              (number first) (working base (numbers items) []) := Evaluates.lookup (by simp [bound, setLocal])
          simpa only [working, numbers, List.map_append, List.map_cons, List.map_nil, List.append_nil] using
            emit base (numbers items) [] [] (number first) (by simpa only [List.append_nil] using looked)
        obtain ⟨finalEnv, rest⟩ := ih (first + 1) (by omega) (items ++ [first]) bound getArray'
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, primeBody,
          Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.filter_cons,
          show decide (Nat.Prime first) = true by simp [prime], if_true, Int.natCast_add,
          Int.natCast_one, List.append_assoc, List.singleton_append] using emitted.append rest
      · have skipped : BlockContinues program bound (primeBody base.length)
            (base ++ [.list (numbers items)]) bound (base ++ [.list (numbers items)]) := by
          have zero : bytes[first].toNat = 0 := by tauto
          exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, zero]) .nil
        obtain ⟨finalEnv, rest⟩ := ih (first + 1) (by omega) items bound getArray'
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, primeBody,
          Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.filter_cons,
          show decide (Nat.Prime first) = false by simp [prime], Int.natCast_add,
          Int.natCast_one] using skipped.append rest

/-- Filtering visits each byte once and allocates one increasing prime list. -/
theorem primes_result (limit : Nat) (large : 2 ≤ limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (bytes : List UInt8) (size : bytes.length = limit)
    (getLimit : env "limit" = some (number limit)) (getArray : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Evaluates program env
      (.listComp (.name "i") [("i", .call (.name "range") [.constant (.int 2), .name "limit"],
        [.subscript (.name "is_prime") (.name "i")])]) heap (.ref heap.length)
      (heap ++ [.list (numbers (primes limit))]) := by
  let inner := comprehensionEnv env [("i", .call (.name "range") [.constant (.int 2), .name "limit"],
    [.subscript (.name "is_prime") (.name "i")])]
  obtain ⟨finalEnv, scanned⟩ := prime_loop 2 (limit - 2) limit (by omega) heap address bytes size found correct [] inner
    (by simpa [inner, comprehensionEnv] using getArray)
  apply Evaluates.list_comp_range (finalEnv := finalEnv) (.cons Evaluates.constant (.cons (Evaluates.lookup getLimit) .nil)) rfl
  simpa only [numbers, List.map_nil, List.nil_append, primes] using scanned

private def sievedHeap (heap : Mutable.Heap) (bytes : List UInt8) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray bytes]
private def primeEnv (env : HeapCalls.Env) (heap : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal env "is_prime" (.ref (heap.length + 1))) "primes" (.ref (heap.length + 2))

/-- The two source assignments retain the sieve and the exact reference prime prefix. -/
theorem prime_setup (limit : Nat) (large : 3 ≤ limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) :
    ∃ bytes, BlockContinues program env ((code.drop 1).take 2) heap (primeEnv env heap)
        (sievedHeap heap bytes ++ [.list (numbers (primes limit))]) ∧ bytes.length = limit ∧
      (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  obtain ⟨bytes, returned, size, correct⟩ := sieve_is_prime.positive_result (limit - 1) (by omega) heap
  let firstEnv := setLocal env "is_prime" (.ref (heap.length + 1))
  have first : BlockContinues program env [code[1]'(by decide)] heap firstEnv (sievedHeap heap bytes) := by
    apply BlockContinues.assign
    apply Evaluates.call (values := [number (limit - 1 : Nat)]) (middle := heap)
      (.cons (Evaluates.binary (av := number limit) (bv := number 1) (middle := heap) (afterRight := heap)
        (Evaluates.lookup getLimit) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, Int.natCast_sub (by omega : 1 ≤ limit)])) .nil)
    exact returned
  have firstLimit : firstEnv "limit" = some (number limit) := by simpa [firstEnv, setLocal] using getLimit
  have firstArray : firstEnv "is_prime" = some (.ref (heap.length + 1)) := by simp [firstEnv, setLocal]
  have size' : bytes.length = limit := by omega
  have second : BlockContinues program firstEnv [code[2]'(by decide)] (sievedHeap heap bytes) (primeEnv env heap)
      (sievedHeap heap bytes ++ [.list (numbers (primes limit))]) := by
    have filtered := primes_result limit (by omega) (sievedHeap heap bytes) firstEnv (heap.length + 1) bytes size'
      firstLimit firstArray (by simp [sievedHeap]) correct
    apply BlockContinues.annAssign
    simpa only [sievedHeap, List.length_append, List.length_cons, List.length_nil, Nat.add_zero] using filtered
  exact ⟨bytes, first.append second, size', correct⟩

private def prefixBody : List Stmt := match (code[5]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []

private theorem prefix_loop (items remaining : List Nat) (position : Nat) (suffix : items.drop position = remaining)
    (base : Mutable.Heap) (address : Nat) (found : base[address]? = some (.list (numbers items)))
    (collected : List Nat) (total : Nat) (env : HeapCalls.Env)
    (getPrefix : env "prefix" = some (.ref base.length)) (getSum : env "s" = some (number total)) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "p" (.ref address) position prefixBody []]
        (working base (numbers collected) []) finalEnv
        (working base (numbers (collected ++ fromTotal total remaining)) []) ∧
      finalEnv "s" = some (number (total + remaining.sum : Nat)) ∧
      ∀ name, name ≠ "p" → name ≠ "s" → finalEnv name = env name := by
  induction remaining generalizing position collected total env with
  | nil =>
      refine ⟨env, ?_, ?_, fun _ _ _ => rfl⟩
      · simp only [fromTotal, List.append_nil]
        apply BlockContinues.sequence_done (items := numbers items)
          (by simp [sequenceValue, Mutable.sequence, working, prefix_found base _ address _ found])
        · have missing : items[position]? = none := by
            simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
          simp [numbers, missing]
        · exact .nil
      · simpa only [List.sum_nil, Nat.add_zero] using getSum
  | cons head tail ih =>
      let bound := setLocal env "p" (number head)
      let ready := setLocal bound "s" (number (total + head : Nat))
      let heap := working base (numbers collected) []
      have getPrefix' : ready "prefix" = some (.ref base.length) := by simpa [ready, bound, setLocal] using getPrefix
      have getSum' : ready "s" = some (number (total + head : Nat)) := by simp [ready, setLocal]
      have updated : BlockContinues program bound [prefixBody[0]'(by decide)] heap ready heap := by
        apply BlockContinues.augAssign (left := number total) (right := number head)
          (by simpa [bound, setLocal] using getSum) (Evaluates.lookup (by simp [bound, setLocal]))
        simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
      have arrayFound : heap[base.length]? = some (.list (numbers collected)) := by simp [heap, ProjectEulerSolutions.HeapCollection.found]
      have value : Evaluates program ready (.name "s") heap (number (total + head : Nat)) heap := Evaluates.lookup getSum'
      have appended : BlockContinues program ready [prefixBody[1]'(by decide)] heap ready
          (working base (numbers (collected ++ [total + head])) []) := by
        have evaluated := Evaluates.append (Evaluates.lookup getPrefix') arrayFound value arrayFound
        apply BlockContinues.discard
        simpa only [heap, numbers, List.map_append, List.map_cons, List.map_nil, store] using evaluated
      obtain ⟨finalEnv, scanned, finalSum, stable⟩ := ih (position + 1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) (collected ++ [total + head]) (total + head) ready getPrefix' getSum'
      refine ⟨finalEnv, ?_, ?_, ?_⟩
      · have selected : items[position]? = some head := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        apply BlockContinues.sequence_step (source := .ref address) (items := numbers items) (value := number head)
          (by simp [sequenceValue, Mutable.sequence, working, prefix_found base _ address _ found]) (by simp [numbers, selected])
        simpa only [Calls.loopContinuation, show Calls.loopControl prefixBody = false from rfl,
          Bool.false_eq_true, if_false, fromTotal, List.append_assoc, List.singleton_append, heap] using
          (updated.append appended).append scanned
      · simpa only [List.sum_cons, Nat.add_assoc] using finalSum
      · intro name notP notS
        exact (stable name notP notS).trans (by simp [ready, bound, setLocal, notP, notS])

/-- The original loop constructs all prefix sums, retaining the prime list and other locals. -/
theorem prefix_setup (items : List Nat) (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env)
    (getPrimes : env "primes" = some (.ref address)) (found : heap[address]? = some (.list (numbers items))) :
    ∃ finalEnv, BlockContinues program env ((code.drop 3).take 3) heap finalEnv
        (heap ++ [.list (numbers (prefixes items))]) ∧
      finalEnv "prefix" = some (.ref heap.length) ∧ finalEnv "s" = some (number items.sum) ∧
      ∀ name, name ≠ "prefix" → name ≠ "s" → name ≠ "p" → finalEnv name = env name := by
  let allocated := setLocal env "prefix" (.ref heap.length)
  let ready := setLocal allocated "s" (number 0)
  have first : BlockContinues program env [code[3]'(by decide)] heap allocated (heap ++ [.list (numbers [0])]) :=
    BlockContinues.annAssign (Evaluates.list (.cons Evaluates.constant .nil))
  have second : BlockContinues program allocated [code[4]'(by decide)] (heap ++ [.list (numbers [0])]) ready
      (heap ++ [.list (numbers [0])]) := BlockContinues.assign Evaluates.constant
  obtain ⟨finalEnv, scanned, finalSum, stable⟩ := prefix_loop items items 0 rfl heap address found [0] 0 ready
    (by simp [ready, allocated, setLocal]) (by simp [ready, setLocal])
  have third : BlockContinues program ready [code[5]'(by decide)] (heap ++ [.list (numbers [0])]) finalEnv
      (heap ++ [.list (numbers (prefixes items))]) := by
    apply BlockContinues.for_sequence (value := .ref address) (items := numbers items)
      (by intros; intro impossible; cases impossible)
      (Evaluates.lookup (by simpa [ready, allocated, setLocal] using getPrimes))
      (by simp [Mutable.sequence, prefix_found heap _ address _ found])
    simpa only [working, prefixes, List.singleton_append] using scanned
  refine ⟨finalEnv, first.append (second.append third), ?_, ?_, ?_⟩
  · exact (stable "prefix" (by decide) (by decide)).trans (by simp [ready, allocated, setLocal])
  · simpa only [Nat.zero_add] using finalSum
  · intro name notPrefix notSum notP
    exact (stable name notP notSum).trans (by simp [ready, allocated, setLocal, notPrefix, notSum])

private theorem read_prefix (items : List Nat) (index address : Nat) (expression : Expr)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (inside : index ≤ items.length)
    (getPrefix : env "prefix" = some (.ref address))
    (found : heap[address]? = some (.list (numbers (prefixes items))))
    (ordinary : ∀ a b c, expression ≠ .slice a b c)
    (evaluated : Evaluates program env expression heap (number index) heap) :
    Evaluates program env (.subscript (.name "prefix") expression) heap (number (items.take index).sum) heap := by
  have mapped : (numbers (prefixes items))[index]? = some (number (items.take index).sum) := by
    simp only [numbers, List.getElem?_map, prefix_get items index inside, Option.map_some]
  have valid : index < (numbers (prefixes items)).length := by simp only [numbers, List.length_map, prefix_length]; omega
  have value : (numbers (prefixes items))[index] = number (items.take index).sum :=
    Option.some.inj ((List.getElem?_eq_getElem valid).symm.trans mapped)
  apply Evaluates.subscript ordinary (Evaluates.lookup getPrefix) evaluated
  simpa only [value] using Mutable.getItem_list heap address index (numbers (prefixes items)) found valid

private def capacityCondition : Expr := match (code[7]? : Option Stmt) with
  | some (.whileDo condition _ _) => condition | _ => .constant .none
private def capacityBody : List Stmt := match (code[7]? : Option Stmt) with
  | some (.whileDo _ body _) => body | _ => []
private def nextIndex : Expr := .binOp (.name "max_len") .add (.constant (.int 1))

/-- The length check prevents evaluating the next prefix entry past the array's end. -/
private theorem capacity_guard (items : List Nat) (limit current address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) (getMaximum : env "max_len" = some (number current))
    (getPrefix : env "prefix" = some (.ref address))
    (found : heap[address]? = some (.list (numbers (prefixes items)))) :
    Evaluates program env capacityCondition heap
      (flag (decide (current < items.length ∧ window items 0 (current + 1) < limit))) heap := by
  have next : Evaluates program env nextIndex heap (number (current + 1 : Nat)) heap :=
    Evaluates.binary (av := number current) (bv := number 1) (Evaluates.lookup getMaximum) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary])
  have sized : Evaluates program env (.builtinCall .len [.name "prefix"]) heap (number (items.length + 1 : Nat)) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getPrefix) .nil)
    simp [primitive, Mutable.length, found, numbers]
  have first : Evaluates program env (.compare nextIndex [(.lt, .builtinCall .len [.name "prefix"])]) heap
      (flag (decide (current < items.length))) heap := by
    apply Evaluates.compare next sized
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases inside : current < items.length
  · have looked := read_prefix items (current + 1) address nextIndex heap env (by omega) getPrefix found
      (by intros; intro impossible; cases impossible) next
    have second : Evaluates program env (.compare (.subscript (.name "prefix") nextIndex) [(.lt, .name "limit")]) heap
        (flag (decide (window items 0 (current + 1) < limit))) heap := by
      apply Evaluates.compare looked (Evaluates.lookup getLimit)
      simp [window, HeapCalls.compare, scalarCompare, evalCmpOp]
    simp only [inside, true_and]
    apply Evaluates.boolean_next (by simpa only [inside, decide_true] using first) rfl rfl
    exact Evaluates.boolean_last (flag := decide (window items 0 (current + 1) < limit)) second rfl
  · simp only [inside, false_and, decide_false]
    apply Evaluates.boolean_stop (by simpa only [inside, decide_false] using first) rfl rfl

/-- Capacity discovery terminates, preserves both tables, and establishes the pruning bound. -/
theorem capacity_result (items : List Nat) (limit current address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (bounded : current ≤ items.length) (getLimit : env "limit" = some (number limit))
    (getMaximum : env "max_len" = some (number current)) (getPrefix : env "prefix" = some (.ref address))
    (found : heap[address]? = some (.list (numbers (prefixes items)))) :
    ∃ maximum finalEnv, BlockContinues program env [code[7]'(by decide)] heap finalEnv heap ∧
      maximum ≤ items.length ∧ finalEnv "max_len" = some (number maximum) ∧
      (maximum = items.length ∨ limit ≤ window items 0 (maximum + 1)) ∧
      ∀ name, name ≠ "max_len" → finalEnv name = env name := by
  induction measure : items.length - current using Nat.strong_induction_on generalizing current env with
  | h measure ih =>
      have tested := capacity_guard items limit current address heap env getLimit getMaximum getPrefix found
      by_cases active : current < items.length ∧ window items 0 (current + 1) < limit
      · let ready := setLocal env "max_len" (number (current + 1 : Nat))
        obtain ⟨maximum, finalEnv, scanned, finalBound, finalMaximum, stopped, stable⟩ :=
          ih (items.length - (current + 1)) (by omega) (current + 1) ready (by omega)
            (by simpa [ready, setLocal] using getLimit) (by simp [ready, setLocal])
            (by simpa [ready, setLocal] using getPrefix) rfl
        refine ⟨maximum, finalEnv, ?_, finalBound, finalMaximum, stopped, ?_⟩
        · apply BlockContinues.while_loop (flag := true) tested (by simp [active, Mutable.truth, scalarTruth])
          have incremented : BlockContinues program env capacityBody heap ready heap := by
            apply BlockContinues.augAssign (left := number current) (right := number 1) getMaximum Evaluates.constant
            simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
          simpa only [Calls.loopContinuation, show Calls.loopControl capacityBody = false from rfl,
            Bool.false_eq_true, if_false] using incremented.append scanned
        · intro name other
          exact (stable name other).trans (by simp [ready, setLocal, other])
      · refine ⟨current, env, ?_, bounded, getMaximum, ?_, fun _ _ => rfl⟩
        · exact BlockContinues.while_loop (flag := false) tested (by simp [active, Mutable.truth, scalarTruth]) .nil
        · by_cases full : current = items.length
          · exact Or.inl full
          · exact Or.inr (by omega)

private structure SearchLocals (env : HeapCalls.Env) (limit maximum count sieveAddress prefixAddress : Nat) : Prop where
  getLimit : env "limit" = some (number limit)
  getMaximum : env "max_len" = some (number maximum)
  getCount : env "n" = some (number count)
  getSieve : env "is_prime" = some (.ref sieveAddress)
  getPrefix : env "prefix" = some (.ref prefixAddress)

def preparedHeap (heap : Mutable.Heap) (limit : Nat) (bytes : List UInt8) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray bytes, .list (numbers (primes limit)), .list (numbers (prefixes (primes limit)))]

/-- All original setup statements establish the native search's arrays, counters, and pruning premise. -/
theorem prepared_result (limit : Nat) (large : 3 ≤ limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) :
    ∃ bytes maximum finalEnv, BlockContinues program env (code.take 9) heap finalEnv (preparedHeap heap limit bytes) ∧
      SearchLocals finalEnv limit maximum (primes limit).length (heap.length + 1) (heap.length + 3) ∧
      bytes.length = limit ∧ (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) ∧
      maximum ≤ (primes limit).length ∧ (∀ candidate, Valid (primes limit) limit candidate → candidate.length ≤ maximum) := by
  obtain ⟨bytes, allocated, size, correct⟩ := prime_setup limit large heap env getLimit
  let base := sievedHeap heap bytes ++ [.list (numbers (primes limit))]
  have primesFound : base[heap.length + 2]? = some (.list (numbers (primes limit))) := by
    simp [base, sievedHeap, List.append_assoc]
  obtain ⟨prefixEnv, prefixed, getPrefix, getSum, prefixStable⟩ := prefix_setup (primes limit) base
    (heap.length + 2) (primeEnv env heap) (by simp [primeEnv, setLocal]) primesFound
  let ready := setLocal prefixEnv "max_len" (number 0)
  have readyLimit : ready "limit" = some (number limit) := by
    rw [show ready "limit" = prefixEnv "limit" by simp [ready, setLocal],
      prefixStable "limit" (by decide) (by decide) (by decide)]
    simpa [primeEnv, setLocal] using getLimit
  have readyPrefix : ready "prefix" = some (.ref base.length) := by simpa [ready, setLocal] using getPrefix
  have readySieve : ready "is_prime" = some (.ref (heap.length + 1)) := by
    rw [show ready "is_prime" = prefixEnv "is_prime" by simp [ready, setLocal],
      prefixStable "is_prime" (by decide) (by decide) (by decide)]
    simp [primeEnv, setLocal]
  have readyPrimes : ready "primes" = some (.ref (heap.length + 2)) := by
    rw [show ready "primes" = prefixEnv "primes" by simp [ready, setLocal],
      prefixStable "primes" (by decide) (by decide) (by decide)]
    simp [primeEnv, setLocal]
  have bufferFound : (base ++ [.list (numbers (prefixes (primes limit)))])[base.length]? =
      some (.list (numbers (prefixes (primes limit)))) := by simp
  obtain ⟨maximum, capacityEnv, capacity, bounded, getMaximum, stopped, stable⟩ := capacity_result (primes limit) limit 0
    base.length (base ++ [.list (numbers (prefixes (primes limit)))]) ready (by omega) readyLimit
    (by simp [ready, setLocal]) readyPrefix bufferFound
  let finalEnv := setLocal capacityEnv "n" (number (primes limit).length)
  have counted : BlockContinues program capacityEnv [code[8]'(by decide)]
      (base ++ [.list (numbers (prefixes (primes limit)))]) finalEnv
      (base ++ [.list (numbers (prefixes (primes limit)))]) := by
    apply BlockContinues.assign
    apply Evaluates.builtin (.cons (Evaluates.lookup ((stable "primes" (by decide)).trans readyPrimes)) .nil)
    simp [primitive, Mutable.length, prefix_found base _ _ _ primesFound, numbers]
  have guard : BlockContinues program env [code[0]'(by decide)] heap env heap := by
    apply BlockContinues.branch (flag := false) (value := flag false)
      (Evaluates.compare (av := number limit) (bv := number 2) (Evaluates.lookup getLimit) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl .nil
  have initialized : BlockContinues program prefixEnv [code[6]'(by decide)]
      (base ++ [.list (numbers (prefixes (primes limit)))]) ready
      (base ++ [.list (numbers (prefixes (primes limit)))]) := BlockContinues.assign Evaluates.constant
  refine ⟨bytes, maximum, finalEnv, ?_, ?_, size, correct, bounded, ?_⟩
  · simpa only [base, sievedHeap, preparedHeap, List.append_assoc] using
      guard.append (allocated.append (prefixed.append (initialized.append (capacity.append counted))))
  · refine ⟨?_, ?_, by simp [finalEnv, setLocal], ?_, ?_⟩
    · exact (show finalEnv "limit" = capacityEnv "limit" from by simp [finalEnv, setLocal]).trans
        ((stable "limit" (by decide)).trans readyLimit)
    · exact (show finalEnv "max_len" = capacityEnv "max_len" from by simp [finalEnv, setLocal]).trans getMaximum
    · exact (show finalEnv "is_prime" = capacityEnv "is_prime" from by simp [finalEnv, setLocal]).trans
        ((stable "is_prime" (by decide)).trans readySieve)
    · have saved := (stable "prefix" (by decide)).trans readyPrefix
      simpa [finalEnv, setLocal, base, sievedHeap] using saved
  · exact maximum_length_bound (primes limit) (primes_ordered limit (by omega)) limit maximum stopped

private def lengthBody : List Stmt := match (code[9]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def rowBody : List Stmt := match (lengthBody[0]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []

private structure SearchHeap (heap : Mutable.Heap) (items : List Nat) (bytes : List UInt8)
    (limit sieveAddress prefixAddress : Nat) : Prop where
  sieve : heap[sieveAddress]? = some (.bytearray bytes)
  prefixes : heap[prefixAddress]? = some (.list (numbers (ProjectEulerSolutions.ConsecutivePrimeSums.prefixes items)))
  size : bytes.length = limit
  correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)

private theorem SearchLocals.set (locals : SearchLocals env limit maximum count sieveAddress prefixAddress)
    (name : String) (value : Mutable.Value) (unused : name ∉ ["limit", "max_len", "n", "is_prime", "prefix"]) :
    SearchLocals (setLocal env name value) limit maximum count sieveAddress prefixAddress := by
  have stable (query : String) (member : query ∈ ["limit", "max_len", "n", "is_prime", "prefix"]) :
      setLocal env name value query = env query := by
    have different : query ≠ name := by intro same; subst query; exact unused member
    simp [setLocal, different]
  exact ⟨(stable "limit" (by simp)).trans locals.getLimit, (stable "max_len" (by simp)).trans locals.getMaximum,
    (stable "n" (by simp)).trans locals.getCount, (stable "is_prime" (by simp)).trans locals.getSieve,
    (stable "prefix" (by simp)).trans locals.getPrefix⟩

private def rowValues (items : List Nat) (answer : Window) : List Mutable.Value :=
  [number (window items answer.first answer.length), number answer.length]

/-- Unsuccessful scans preserve the heap; successful ones return one fresh pair. -/
private def Scan (items : List Nat) (invariant : HeapCalls.Env → Prop) (env : HeapCalls.Env)
    (body : List Stmt) (heap : Mutable.Heap) : Option Window → Prop
  | none => ∃ finalEnv, BlockContinues program env body heap finalEnv heap ∧ invariant finalEnv
  | some answer => ∀ following,
      BlockReturns program env (body ++ following) heap (.ref heap.length) (heap ++ [.tuple (rowValues items answer)])

/-- Both reads stay in bounds, and Python's signed subtraction equals the natural window sum. -/
private theorem window_result (items : List Nat) (first length prefixAddress : Nat) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (inside : first + length ≤ items.length)
    (getI : env "i" = some (number first)) (getLength : env "length" = some (number length))
    (getPrefix : env "prefix" = some (.ref prefixAddress))
    (found : heap[prefixAddress]? = some (.list (numbers (prefixes items)))) :
    Evaluates program env (.binOp (.subscript (.name "prefix") (.binOp (.name "i") .add (.name "length")))
      .sub (.subscript (.name "prefix") (.name "i"))) heap (number (window items first length)) heap := by
  have endIndex : Evaluates program env (.binOp (.name "i") .add (.name "length")) heap
      (number (first + length : Nat)) heap :=
    Evaluates.binary (av := number first) (bv := number length) (Evaluates.lookup getI) (Evaluates.lookup getLength)
      (by simp [Mutable.binary, scalarBinary])
  have right := read_prefix items (first + length) prefixAddress (.binOp (.name "i") .add (.name "length")) heap env
    inside getPrefix found (by intros; intro impossible; cases impossible) endIndex
  have left := read_prefix items first prefixAddress (.name "i") heap env (by omega) getPrefix found
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getI)
  have increasing : (items.take first).sum ≤ (items.take (first + length)).sum := by
    simpa [window] using window_mono_length items 0 first (first + length) (by omega)
  have naturalDifference : (items.take (first + length)).sum - (items.take first).sum = window items first length := by
    simpa only [prefix_get items (first + length) inside, prefix_get items first (by omega), Option.getD_some] using
      prefix_difference items first length inside
  apply Evaluates.binary right left
  simp [Mutable.binary, scalarBinary, ← Int.natCast_sub increasing, naturalDifference]

private theorem large_test (limit value : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) (getValue : env "val" = some (number value)) :
    Evaluates program env (.compare (.name "val") [(.gtE, .name "limit")]) heap
      (flag (decide (limit ≤ value))) heap :=
  Evaluates.compare (Evaluates.lookup getValue) (Evaluates.lookup getLimit)
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

private theorem prime_test (items : List Nat) (bytes : List UInt8) (limit maximum sieveAddress prefixAddress value : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (inside : value < limit)
    (snapshot : SearchHeap heap items bytes limit sieveAddress prefixAddress)
    (locals : SearchLocals env limit maximum items.length sieveAddress prefixAddress)
    (getValue : env "val" = some (number value)) :
    ∃ byte : UInt8, Evaluates program env (.subscript (.name "is_prime") (.name "val")) heap (number byte.toNat) heap ∧
      (byte.toNat ≠ 0 ↔ Nat.Prime value) := by
  have valid : value < bytes.length := by rw [snapshot.size]; exact inside
  refine ⟨bytes[value], ?_, snapshot.correct value bytes[value] (List.getElem?_eq_getElem valid)⟩
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup locals.getSieve) (Evaluates.lookup getValue)
  exact Mutable.getItem_buffer heap sieveAddress value bytes snapshot.sieve valid

private theorem row_return (items : List Nat) (bytes : List UInt8) (limit maximum sieveAddress prefixAddress first length : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (inside : first + length ≤ items.length)
    (snapshot : SearchHeap heap items bytes limit sieveAddress prefixAddress)
    (locals : SearchLocals env limit maximum items.length sieveAddress prefixAddress)
    (getI : env "i" = some (number first)) (getLength : env "length" = some (number length))
    (below : window items first length < limit) (prime : Nat.Prime (window items first length)) (following : List Stmt) :
    BlockReturns program env (rowBody ++ following) heap (.ref heap.length)
      (heap ++ [.tuple (rowValues items ⟨first, length⟩)]) := by
  let ready := setLocal env "val" (number (window items first length))
  have readyLocals := locals.set "val" (number (window items first length)) (by simp)
  have getValue : ready "val" = some (number (window items first length)) := by simp [ready, setLocal]
  obtain ⟨byte, tested, live⟩ := prime_test items bytes limit maximum sieveAddress prefixAddress
    (window items first length) heap ready below snapshot readyLocals getValue
  apply BlockReturns.assign (window_result items first length prefixAddress heap env inside getI getLength locals.getPrefix snapshot.prefixes)
  apply BlockReturns.branch (flag := false)
    (large_test limit (window items first length) heap ready readyLocals.getLimit getValue)
    (by simp [show ¬ limit ≤ window items first length by omega, Mutable.truth, scalarTruth])
  apply BlockReturns.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, live.mpr prime])
  apply BlockReturns.return_value
  apply Evaluates.tuple (values := [number (window items first length), number length])
    (.cons (Evaluates.lookup getValue) (.cons (Evaluates.lookup (by simpa [ready, setLocal] using getLength)) .nil))
    (by simp)

private theorem row_skip (items : List Nat) (bytes : List UInt8) (limit maximum sieveAddress prefixAddress first length : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (inside : first + length ≤ items.length)
    (snapshot : SearchHeap heap items bytes limit sieveAddress prefixAddress)
    (locals : SearchLocals env limit maximum items.length sieveAddress prefixAddress)
    (getI : env "i" = some (number first)) (getLength : env "length" = some (number length))
    (below : window items first length < limit) (composite : ¬ Nat.Prime (window items first length)) :
    BlockContinues program env rowBody heap (setLocal env "val" (number (window items first length))) heap := by
  let ready := setLocal env "val" (number (window items first length))
  have readyLocals := locals.set "val" (number (window items first length)) (by simp)
  have getValue : ready "val" = some (number (window items first length)) := by simp [ready, setLocal]
  obtain ⟨byte, tested, live⟩ := prime_test items bytes limit maximum sieveAddress prefixAddress
    (window items first length) heap ready below snapshot readyLocals getValue
  have assigned : BlockContinues program env [rowBody[0]'(by decide)] heap ready heap :=
    BlockContinues.assign (window_result items first length prefixAddress heap env inside getI getLength locals.getPrefix snapshot.prefixes)
  have guarded : BlockContinues program ready [rowBody[1]'(by decide)] heap ready heap :=
    BlockContinues.branch (flag := false)
      (large_test limit (window items first length) heap ready readyLocals.getLimit getValue)
      (by simp [show ¬ limit ≤ window items first length by omega, Mutable.truth, scalarTruth]) .nil
  have skipped : BlockContinues program ready [rowBody[2]'(by decide)] heap ready heap := by
    have zero : byte.toNat = 0 := by tauto
    exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, zero]) .nil
  exact assigned.append (guarded.append skipped)

private theorem row_break (items : List Nat) (limit maximum sieveAddress prefixAddress first length : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (inside : first + length ≤ items.length)
    (found : heap[prefixAddress]? = some (.list (numbers (prefixes items))))
    (locals : SearchLocals env limit maximum items.length sieveAddress prefixAddress)
    (getI : env "i" = some (number first)) (getLength : env "length" = some (number length))
    (tooLarge : limit ≤ window items first length) (next : Stmt) :
    BlockContinues program env (rowBody ++ [.loopResume next]) heap
      (setLocal env "val" (number (window items first length))) heap := by
  let ready := setLocal env "val" (number (window items first length))
  have readyLocals := locals.set "val" (number (window items first length)) (by simp)
  have getValue : ready "val" = some (number (window items first length)) := by simp [ready, setLocal]
  have assigned : BlockContinues program env [rowBody[0]'(by decide)] heap ready heap :=
    BlockContinues.assign (window_result items first length prefixAddress heap env inside getI getLength locals.getPrefix found)
  apply assigned.append
  apply BlockContinues.branch_next (flag := true)
    (large_test limit (window items first length) heap ready readyLocals.getLimit getValue)
    (by simp [tooLarge, Mutable.truth, scalarTruth])
  exact BlockContinues.break_loop (next := next) (rest := []) rfl .nil

private theorem scan_weaken {answer : Option Window}
    (implication : ∀ finalEnv, invariant finalEnv → weaker finalEnv)
    (executed : Scan items invariant env body heap answer) : Scan items weaker env body heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, continued, valid⟩ := executed
      exact ⟨finalEnv, continued, implication finalEnv valid⟩
  | some answer => exact executed

private theorem scan_prepend {answer : Option Window}
    (continued : BlockContinues program env body heap middleEnv heap)
    (rest : Scan items invariant middleEnv following heap answer) :
    Scan items invariant env (body ++ following) heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finished, valid⟩ := rest
      exact ⟨finalEnv, continued.append finished, valid⟩
  | some answer =>
      intro after
      simpa only [List.append_assoc] using continued.returns (rest after)

private theorem scan_resume {answer : Option Window}
    (rest : Scan items invariant env [next] heap answer) :
    Scan items invariant env [.loopResume next] heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finished, valid⟩ := rest
      exact ⟨finalEnv, fun following anchors stack => .next rfl (finished following anchors stack), valid⟩
  | some answer =>
      intro following
      exact BlockReturns.loop_resume (rest following)

private theorem scan_range_step {answer : Option Window}
    (active : Calls.rangeContinues first stop stride = true)
    (rest : Scan items invariant (setLocal env name (number first))
      (body ++ [Calls.loopContinuation body (.rangeStepLoop name (first + stride) stop stride body [])]) heap answer) :
    Scan items invariant env [.rangeStepLoop name first stop stride body []] heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finished, valid⟩ := rest
      exact ⟨finalEnv, BlockContinues.range_step active finished, valid⟩
  | some answer =>
      intro following
      apply BlockReturns.range_step active
      simpa only [List.append_assoc] using rest following

private theorem scan_for_range {answer : Option Window}
    (arguments : Evaluates.Arguments program env expressions heap values heap)
    (bounds : rangeBounds values = .ok (first, stop, stride))
    (rest : Scan items invariant env [.rangeStepLoop name first stop stride body []] heap answer) :
    Scan items invariant env [.forIn (.name name) (.call (.name "range") expressions) body []] heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finished, valid⟩ := rest
      exact ⟨finalEnv, BlockContinues.for_range arguments bounds finished, valid⟩
  | some answer =>
      intro following
      exact BlockReturns.for_range arguments bounds (rest following)

/-- The live inner range agrees with the native row scan, including its early break and return. -/
theorem inner_loop (items : List Nat) (bytes : List UInt8) (limit maximum sieveAddress prefixAddress length first count stop : Nat)
    (endpoint : first + count = stop) (bounded : stop + length ≤ items.length + 1)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (snapshot : SearchHeap heap items bytes limit sieveAddress prefixAddress)
    (locals : SearchLocals env limit maximum items.length sieveAddress prefixAddress)
    (getLength : env "length" = some (number length)) :
    Scan items (fun finalEnv => SearchLocals finalEnv limit maximum items.length sieveAddress prefixAddress ∧
      finalEnv "length" = some (number length)) env [.rangeStepLoop "i" first stop 1 rowBody []] heap
      (scanRow items limit length first count) := by
  induction count generalizing first env with
  | zero =>
      refine ⟨env, ?_, locals, getLength⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "i" (number first)
      let ready := setLocal bound "val" (number (window items first length))
      have boundLocals : SearchLocals bound limit maximum items.length sieveAddress prefixAddress :=
        locals.set "i" (number first) (by simp)
      have readyLocals : SearchLocals ready limit maximum items.length sieveAddress prefixAddress :=
        boundLocals.set "val" (number (window items first length)) (by simp)
      have boundLength : bound "length" = some (number length) := by simpa [bound, setLocal] using getLength
      have readyLength : ready "length" = some (number length) := by simpa [ready, setLocal] using boundLength
      have inside : first + length ≤ items.length := by omega
      have active : Calls.rangeContinues first stop 1 = true := by simp [Calls.rangeContinues]; omega
      by_cases tooLarge : limit ≤ window items first length
      · simp only [scanRow, if_pos tooLarge]
        refine ⟨ready, ?_, readyLocals, readyLength⟩
        apply BlockContinues.range_step active
        simpa only [Calls.loopContinuation, show Calls.loopControl rowBody = true from rfl,
          if_true, Int.natCast_add, Int.natCast_one] using
          row_break items limit maximum sieveAddress prefixAddress first length heap bound inside snapshot.prefixes
            boundLocals (by simp [bound, setLocal]) boundLength tooLarge
            (.rangeStepLoop "i" ((first + 1 : Nat) : Int) stop 1 rowBody [])
      · by_cases prime : Nat.Prime (window items first length)
        · simp only [scanRow, if_neg tooLarge, if_pos prime]
          intro following
          apply BlockReturns.range_step active
          simpa only [List.append_assoc] using row_return items bytes limit maximum sieveAddress prefixAddress first length
            heap bound inside snapshot boundLocals (by simp [bound, setLocal]) boundLength (by omega) prime
            ([Calls.loopContinuation rowBody (.rangeStepLoop "i" ((first : Int) + 1) stop 1 rowBody [])] ++ following)
        · simp only [scanRow, if_neg tooLarge, if_neg prime]
          have continued := row_skip items bytes limit maximum sieveAddress prefixAddress first length heap bound inside
            snapshot boundLocals (by simp [bound, setLocal]) boundLength (by omega) prime
          have rest := ih (first + 1) (by omega) ready readyLocals readyLength
          have resumed := scan_resume rest
          apply scan_range_step active
          simpa only [Calls.loopContinuation, show Calls.loopControl rowBody = true from rfl,
            if_true, Int.natCast_add, Int.natCast_one] using scan_prepend continued resumed

/-- The source row bounds enumerate all starts whose fixed-width window fits in the prime list. -/
theorem row_result (items : List Nat) (bytes : List UInt8) (limit maximum sieveAddress prefixAddress length : Nat)
    (bounded : length ≤ items.length) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (snapshot : SearchHeap heap items bytes limit sieveAddress prefixAddress)
    (locals : SearchLocals env limit maximum items.length sieveAddress prefixAddress)
    (getLength : env "length" = some (number length)) :
    Scan items (fun finalEnv => SearchLocals finalEnv limit maximum items.length sieveAddress prefixAddress ∧
      finalEnv "length" = some (number length)) env lengthBody heap
      (scanRow items limit length 0 (items.length - length + 1)) := by
  have rest := inner_loop items bytes limit maximum sieveAddress prefixAddress length 0 (items.length - length + 1)
    (items.length - length + 1) (by omega) (by omega) heap env snapshot locals getLength
  apply scan_for_range (values := [number 0, number (items.length - length + 1 : Nat)])
    (first := 0) (stop := (items.length - length + 1 : Nat)) (stride := 1)
  · refine .cons Evaluates.constant (.cons ?_ .nil)
    apply Evaluates.binary (av := number (items.length - length : Nat)) (bv := number 1)
      (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (av := number items.length) (bv := number length) (middle := heap) (afterRight := heap)
        (Evaluates.lookup locals.getCount) (Evaluates.lookup getLength)
        (by simp [Mutable.binary, scalarBinary, Int.natCast_sub bounded])
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary]
  · rfl
  · exact rest

/-- Descending source lengths select precisely the native search's first successful row. -/
theorem outer_loop (items : List Nat) (bytes : List UInt8) (limit maximum sieveAddress prefixAddress current : Nat)
    (bounded : current ≤ items.length) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (snapshot : SearchHeap heap items bytes limit sieveAddress prefixAddress)
    (locals : SearchLocals env limit maximum items.length sieveAddress prefixAddress) :
    Scan items (fun finalEnv => SearchLocals finalEnv limit maximum items.length sieveAddress prefixAddress) env
      [.rangeStepLoop "length" current 0 (-1) lengthBody []] heap (scanLengths items limit current) := by
  induction current generalizing env with
  | zero =>
      refine ⟨env, ?_, locals⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil
  | succ current ih =>
      let bound := setLocal env "length" (number (current + 1 : Nat))
      have boundLocals : SearchLocals bound limit maximum items.length sieveAddress prefixAddress :=
        locals.set "length" (number (current + 1 : Nat)) (by simp)
      have scanned := row_result items bytes limit maximum sieveAddress prefixAddress (current + 1) bounded heap bound
        snapshot boundLocals (by simp [bound, setLocal])
      have active : Calls.rangeContinues ((current + 1 : Nat) : Int) 0 (-1) = true := by simp [Calls.rangeContinues]
      cases rowAnswer : scanRow items limit (current + 1) 0 (items.length - (current + 1) + 1) with
      | none =>
          rw [rowAnswer] at scanned
          obtain ⟨rowEnv, continued, rowLocals, getLength⟩ := scanned
          have rest := ih (by omega) rowEnv rowLocals
          simp only [scanLengths, rowAnswer, Option.orElse_none]
          apply scan_range_step active
          have previous : (((current + 1 : Nat) : Int) + -1) = (current : Int) := by omega
          simpa only [Calls.loopContinuation, show Calls.loopControl lengthBody = false from rfl,
            Bool.false_eq_true, if_false, previous] using scan_prepend continued rest
      | some answer =>
          rw [rowAnswer] at scanned
          simp only [scanLengths, rowAnswer, Option.orElse_some]
          intro following
          apply BlockReturns.range_step active
          simpa only [List.append_assoc] using scanned
            ([Calls.loopContinuation lengthBody (.rangeStepLoop "length" (((current + 1 : Nat) : Int) + -1) 0 (-1) lengthBody [])] ++ following)

theorem search_result (items : List Nat) (bytes : List UInt8) (limit maximum sieveAddress prefixAddress : Nat)
    (bounded : maximum ≤ items.length) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (snapshot : SearchHeap heap items bytes limit sieveAddress prefixAddress)
    (locals : SearchLocals env limit maximum items.length sieveAddress prefixAddress) :
    Scan items (fun finalEnv => SearchLocals finalEnv limit maximum items.length sieveAddress prefixAddress) env
      [code[9]'(by decide)] heap (scanLengths items limit maximum) := by
  apply scan_for_range (values := [number maximum, number 0, number (-1)])
    (first := maximum) (stop := 0) (stride := -1)
    (.cons (Evaluates.lookup locals.getMaximum) (.cons Evaluates.constant
      (.cons (Evaluates.unary Evaluates.constant rfl) .nil))) rfl
  exact outer_loop items bytes limit maximum sieveAddress prefixAddress maximum bounded heap env snapshot locals

def answerPair (limit : Nat) : Nat × Nat := match reference (primes limit) limit with
  | none => (0, 0)
  | some chosen => (window (primes limit) chosen.first chosen.length, chosen.length)

private def pairValues (answer : Nat × Nat) : List Mutable.Value := [number answer.1, number answer.2]

/-- The entire original helper returns the same pair as exhaustive mathematical enumeration. -/
theorem positive_result (limit : Nat) (large : 3 ≤ limit) (heap : Mutable.Heap) :
    ∃ bytes, ReturnsValue program "longest_consecutive_prime_sum" [number limit] heap
      (.ref (preparedHeap heap limit bytes).length)
      (preparedHeap heap limit bytes ++ [.tuple (pairValues (answerPair limit))]) := by
  let env := functionEnv ((findFunctionIn program.body "longest_consecutive_prime_sum").getD ([], [])).1
    ((findFunctionIn program.body "longest_consecutive_prime_sum").getD ([], [])).2 [number limit] []
  obtain ⟨bytes, maximum, ready, prepared, locals, size, correct, bounded, covers⟩ := prepared_result limit large heap env rfl
  let base := preparedHeap heap limit bytes
  have snapshot : SearchHeap base (primes limit) bytes limit (heap.length + 1) (heap.length + 3) :=
    ⟨by simp [base, preparedHeap], by simp [base, preparedHeap], size, correct⟩
  have scanned := search_result (primes limit) bytes limit maximum (heap.length + 1) (heap.length + 3)
    bounded base ready snapshot locals
  have same := scanLengths_eq_reference (primes limit) (primes_ordered limit (by omega)) limit maximum bounded covers
  rw [same] at scanned
  refine ⟨bytes, ?_⟩
  apply BlockReturns.function rfl rfl
  cases selected : reference (primes limit) limit with
  | none =>
      rw [selected] at scanned
      obtain ⟨finalEnv, continued, finalLocals⟩ := scanned
      have finished : BlockReturns program finalEnv (code.drop 10) base (.ref base.length)
          (base ++ [.tuple (pairValues (answerPair limit))]) := by
        simp only [answerPair, selected]
        apply BlockReturns.return_value
        exact Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp [pairValues])
      have returned := continued.returns finished
      simpa only [List.take_append_drop] using prepared.returns returned
  | some chosen =>
      rw [selected] at scanned
      have returned := scanned (code.drop 10)
      have joined := prepared.returns returned
      simpa only [answerPair, selected, pairValues, rowValues, List.take_append_drop] using joined

theorem small_result (limit : Int) (small : limit ≤ 2) (heap : Mutable.Heap) :
    ReturnsValue program "longest_consecutive_prime_sum" [number limit] heap (.ref heap.length)
      (heap ++ [.tuple (pairValues (0, 0))]) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := heap)
    (Evaluates.compare (av := number limit) (bv := number 2) (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])) rfl
  apply BlockReturns.return_value
  exact Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp [pairValues])

/-- The helper terminates for every signed bound and returns its complete reference pair. -/
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ address extra, ReturnsValue program "longest_consecutive_prime_sum" [number limit] heap (.ref address) (heap ++ extra) ∧
      (heap ++ extra)[address]? = some (.tuple (pairValues (answerPair limit.toNat))) := by
  by_cases small : limit ≤ 2
  · have answer : answerPair limit.toNat = (0, 0) := by
      have empty : limit.toNat - 2 = 0 := by omega
      simp [answerPair, primes, empty, reference, candidates, row, Valid]
    refine ⟨heap.length, [.tuple (pairValues (0, 0))], small_result limit small heap, ?_⟩
    simp [answer]
  · lift limit to Nat using (by omega : 0 ≤ limit)
    obtain ⟨bytes, returned⟩ := positive_result limit (by omega) heap
    refine ⟨(preparedHeap heap limit bytes).length,
      [.bytearray [1], .bytearray bytes, .list (numbers (primes limit)), .list (numbers (prefixes (primes limit))),
        .tuple (pairValues (answerPair limit))], ?_, ?_⟩
    · simpa only [preparedHeap, List.append_assoc, List.cons_append, List.nil_append] using returned
    · simp [preparedHeap]

end longest_consecutive_prime_sum

namespace solve
/-- The public solver reads the first field of the helper's private result tuple. -/
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "solve" [number limit] heap
      (number (longest_consecutive_prime_sum.answerPair limit.toNat).1) (heap ++ extra) := by
  obtain ⟨address, extra, returned, found⟩ := longest_consecutive_prime_sum.result limit heap
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.subscript (middle := heap ++ extra) (afterIndex := heap ++ extra)
    (by intros; intro impossible; cases impossible)
    (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) returned) Evaluates.constant
  simp [Mutable.getItem, found, longest_consecutive_prime_sum.pairValues, Mutable.index, Except.map]

theorem «Returns» (limit : Int) : (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨extra, returned⟩ := result limit []
  exact returned.scalar_returns

theorem «Supported» (limit : Int) : (HeapCalls.scalarFunction program "solve" [.int limit]).Supported := («Returns» limit).supported
end solve
end solvers.«50»
