import «.formalsw».solvers.«50.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.PrimeSieve
import ProjectEulerSolutions.Lib.BooleanPrimeSieve

set_option Elab.async false

namespace solvers.«50»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P50Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

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
end solvers.«50»
