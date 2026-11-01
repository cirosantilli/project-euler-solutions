import «.formalsw».solvers.«87.py.ast»
import ProjectEulerStatements.Lib.PrimePartitions
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.PrimeSieve

/-! The inclusive bytearray sieve marks exactly the source's extended slices. -/
namespace solvers.«87»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.PrimeSieve
set_option Elab.async false
abbrev program := Solvers.P87Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev encode (items : List Nat) : List Mutable.Value := items.map number
abbrev integer (n : Int) : Mutable.Value := .scalar (.int n)

namespace sieve_primes
private abbrev code : List Stmt := executableBody
  ((findFunctionIn program.body "sieve_primes").getD ([], [])).2
private def loopBody : List Stmt := match (code[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []
private def markBody : List Stmt := match (loopBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []
private theorem heap_inside {heap : Mutable.Heap} {address : Nat}
    (found : heap[address]? = some object) : address < heap.length :=
  (List.getElem?_eq_some_iff.mp found).1

private theorem found_append {heap : Mutable.Heap} {address : Nat}
    (found : heap[address]? = some (object : Mutable.Object)) (extra : Mutable.Heap) :
    (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (heap_inside found)]
  exact found

private theorem heap_set_same (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  have entry := (List.getElem?_eq_some_iff.mp found).2
  rw [← entry]
  exact List.set_getElem_self (heap_inside found)

private theorem initial_slice (limit : Nat) (large : 2 ≤ limit) :
    FormalSW.Py.Builtin.ByteArray.setSlice (List.replicate (limit+1) 1)
      { start := some 0, stop := some 2 } [0, 0] = .ok (initial limit) := by
  have normalized : FormalSW.Py.Builtin.Slice.indices { start := some 0, stop := some 2 } (limit+1) =
      .ok (0, 2, 1) := by
    change Except.ok (max (0 : Int) (min 0 (limit+1)), max (0 : Int) (min 2 (limit+1)), (1 : Int)) = _
    simp [min_eq_left (show (2 : Int) ≤ limit+1 by omega)]
  simp only [FormalSW.Py.Builtin.ByteArray.setSlice, List.isEmpty_cons, Bool.false_eq_true, if_false]
  unfold FormalSW.Py.Builtin.List.setSlice
  rw [show (List.replicate (limit+1) (1 : UInt8)).length = limit+1 by simp, normalized]
  change Except.ok ((List.replicate (limit+1) (1 : UInt8)).take 0 ++ [0,0] ++
    (List.replicate (limit+1) (1 : UInt8)).drop 2) = _
  simp [initial]

private theorem mark_buffer (limit p : Nat) (bytes : List UInt8) (heap : Mutable.Heap) (address : Nat)
    (lower : 2 ≤ p) (size : bytes.length = limit+1) (inside : p*p ≤ limit)
    (found : heap[address]? = some (.bytearray bytes)) :
    Mutable.setSlice (.ref address) (number (p*p)) (number (limit+1)) (number p)
      (.scalar (.bytes (List.replicate (((limit : Int)-p*p)/p+1).toNat 0))) heap =
      some (.ok (.scalar .none), heap.set address (.bytearray (mark bytes p))) := by
  apply Mutable.setSlice_buffer heap address bytes _ _ _ _ _
    { start := some (p*p), stop := some (limit+1), step := some p } found
  · simpa only [number, Nat.cast_mul, Nat.cast_add, Nat.cast_one] using
      Mutable.sliceBounds_ints (p*p) (limit+1) p heap (by omega)
  · have pure := mark_slice bytes p lower (by omega)
    have count : ((bytes.length : Int)-p*p-1)/p+1 = ((limit : Int)-p*p)/p+1 := by
      rw [size]
      congr 2
      push_cast
      ring
    rw [count] at pure
    simpa only [size, Nat.cast_add, Nat.cast_one] using pure

private def markedEnv (env : HeapCalls.Env) (p : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "step" (number p)) "start" (number (p*p))

private theorem mark_result (limit p : Nat) (bytes : List UInt8) (heap : Mutable.Heap) (address : Nat)
    (env : HeapCalls.Env) (lower : 2 ≤ p) (size : bytes.length = limit+1) (inside : p*p ≤ limit)
    (getP : env "p" = some (number p)) (getLimit : env "n" = some (number limit))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    BlockContinues program env markBody heap (markedEnv env p)
      (heap.set address (.bytearray (mark bytes p))) := by
  let e1 := setLocal env "step" (number p)
  let e2 := markedEnv env p
  have first : BlockContinues program env [markBody[0]'(by decide)] heap e1 heap :=
    BlockContinues.assign (Evaluates.lookup getP)
  have second : BlockContinues program e1 [markBody[1]'(by decide)] heap e2 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup (by simpa [e1, setLocal] using getP))
      (Evaluates.lookup (by simpa [e1, setLocal] using getP))
      (by simp [Mutable.binary, scalarBinary, number])
  have getStep : e2 "step" = some (number p) := by simp [e2, markedEnv, setLocal]
  have getStart : e2 "start" = some (number (p*p)) := by simp [e2, markedEnv, setLocal]
  have getLimit' : e2 "n" = some (number limit) := by simpa [e2, markedEnv, setLocal] using getLimit
  have count : Evaluates program e2
      (.binOp (.binOp (.binOp (.name "n") .sub (.name "start")) .floorDiv (.name "step"))
        .add (.constant (.int 1))) heap (integer (((limit : Int)-p*p)/p+1)) heap := by
    apply Evaluates.binary (av := integer (((limit : Int)-p*p)/p)) (bv := number 1)
      (middle := heap) (afterRight := heap)
    · apply Evaluates.binary (av := integer ((limit : Int)-p*p)) (bv := number p)
        (middle := heap) (afterRight := heap)
      · exact Evaluates.binary (Evaluates.lookup getLimit') (Evaluates.lookup getStart)
          (by simp [Mutable.binary, scalarBinary, integer])
      · exact Evaluates.lookup getStep
      · simp [Mutable.binary, scalarBinary, integer, show p ≠ 0 by omega, show 0 < p by omega]
    · exact Evaluates.constant
    · rfl
  have third : BlockContinues program e2 [markBody[2]'(by decide)] heap e2
      (heap.set address (.bytearray (mark bytes p))) := by
    apply BlockContinues.setSlice (afterRhs := heap) (afterTarget := heap)
      (receiver := .ref address) (lowerValue := number (p*p)) (upperValue := number (limit+1))
      (strideValue := number p) (value := .scalar (.bytes (List.replicate (((limit : Int)-p*p)/p+1).toNat 0)))
    · exact Evaluates.binary Evaluates.constant count (by simp [Mutable.binary, scalarBinary])
    · exact .cons (Evaluates.lookup (by simpa [e2, markedEnv, setLocal] using getArray))
        (.cons (Evaluates.lookup getStart) (.cons
          (Evaluates.binary (Evaluates.lookup getLimit') Evaluates.constant
            (by simp [Mutable.binary, scalarBinary, number])) (.cons (Evaluates.lookup getStep) .nil)))
    · exact mark_buffer limit p bytes heap address lower size inside found
  exact first.append (second.append third)

private def initialHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray (initial limit)]

private theorem setup (limit : Nat) (large : 2 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "n" = some (number limit)) :
    BlockContinues program env ((code.drop 1).take 3) heap
      (setLocal (setLocal env "is_prime" (.ref (heap.length+1))) "limit" (number (Nat.sqrt limit))) (initialHeap heap limit) := by
  let ready := setLocal env "is_prime" (.ref (heap.length+1))
  let allocated := heap ++ [.bytearray [1], .bytearray (List.replicate (limit+1) 1)]
  have first : BlockContinues program env [code[1]'(by decide)] heap ready allocated := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := .ref heap.length) (bv := number (limit+1))
      (middle := heap ++ [.bytearray [1]]) (afterRight := heap ++ [.bytearray [1]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
    · simp [allocated, Mutable.binary, Mutable.allocateBuffer, Except.map, show (BinOp.mult == .mult) = true from rfl]
  have second : BlockContinues program ready [code[2]'(by decide)] allocated ready (initialHeap heap limit) := by
    apply BlockContinues.setSlice (value := .scalar (.bytes [0,0])) (afterRhs := allocated) (afterTarget := allocated)
      (receiver := .ref (heap.length+1)) (lowerValue := number 0) (upperValue := number 2) (strideValue := .scalar .none)
    · exact Evaluates.constant
    · exact .cons (Evaluates.lookup rfl) (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))
    · have write := Mutable.setSlice_buffer allocated (heap.length+1) (List.replicate (limit+1) 1)
        [0,0] (initial limit) (number 0) (number 2) (.scalar .none)
        { start := some 0, stop := some 2 } (by simp [allocated]) rfl (initial_slice limit large)
      simpa [allocated, initialHeap] using write
  have root : Evaluates program ready (.call (.name "isqrt") [.name "n"]) (initialHeap heap limit)
      (number (Nat.sqrt limit)) (initialHeap heap limit) := by
    apply Evaluates.library_call (.cons (Evaluates.lookup (by simpa [ready, setLocal] using getLimit)) .nil) rfl
    change scalarResult (some (Math.isqrtCall [.int limit])) (initialHeap heap limit) = _
    have same : Math.isqrt limit = Nat.sqrt limit := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
    simp [scalarResult, Math.isqrtCall, same, number, show ¬ (limit : Int) < 0 by omega]
  exact first.append (second.append (BlockContinues.assign root))

private def Stable (before after : HeapCalls.Env) : Prop :=
  after "n" = before "n" ∧ after "is_prime" = before "is_prime"

private theorem read_buffer (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat)
    (bytes : List UInt8) (valid : p < bytes.length) (getP : env "p" = some (number p))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    Evaluates program env (.subscript (.name "is_prime") (.name "p")) heap (number bytes[p].toNat) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; simp) (Evaluates.lookup getArray) (Evaluates.lookup getP)
  exact Mutable.getItem_buffer heap address p bytes found valid

private theorem iteration (limit p : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (lower : 2 ≤ p) (inside : p*p ≤ limit)
    (getP : env "p" = some (number p)) (getLimit : env "n" = some (number limit))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes))
    (sieved : Sieved limit p bytes) :
    ∃ finalEnv finalBytes, BlockContinues program env loopBody heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Sieved limit (p+1) finalBytes ∧ Stable env finalEnv := by
  have size := sieved.1
  have valid : p < bytes.length := by nlinarith
  have live : bytes[p].toNat ≠ 0 ↔ Nat.Prime p :=
    (sieved.2 p bytes[p] lower (List.getElem?_eq_getElem valid)).trans (survives_current p lower)
  by_cases prime : Nat.Prime p
  · refine ⟨markedEnv env p, mark bytes p, ?_, sieved_mark limit p bytes sieved prime, ?_⟩
    · apply BlockContinues.branch (flag := true) (read_buffer heap env address p bytes valid getP getArray found)
        (by simp [Mutable.truth, scalarTruth, live.mpr prime])
      exact mark_result limit p bytes heap address env lower size inside getP getLimit getArray found
    · simp [Stable, markedEnv, setLocal]
  · refine ⟨env, bytes, ?_, sieved_skip limit p bytes sieved prime, rfl, rfl⟩
    rw [heap_set_same heap address _ found]
    apply BlockContinues.branch (flag := false) (read_buffer heap env address p bytes valid getP getArray found)
    · have zero : bytes[p].toNat = 0 := by tauto
      simp [Mutable.truth, scalarTruth, zero]
    · exact .nil

private theorem loop (limit p : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (lower : 2 ≤ p)
    (getLimit : env "n" = some (number limit))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes))
    (sieved : Sieved limit p bytes) :
    ∃ finalEnv finalBytes next,
      BlockContinues program env [.rangeStepLoop "p" p (Nat.sqrt limit + 1) 1 loopBody []] heap finalEnv
        (heap.set address (.bytearray finalBytes)) ∧ Sieved limit next finalBytes ∧
      limit < next^2 ∧ Stable env finalEnv := by
  by_cases inside : p ≤ Nat.sqrt limit
  · let entered := setLocal env "p" (number p)
    have enteredN : entered "n" = some (number limit) := by simpa [entered, setLocal] using getLimit
    have enteredArray : entered "is_prime" = some (.ref address) := by simpa [entered, setLocal] using getArray
    obtain ⟨middleEnv, middleBytes, iterated, nextSieved, stable⟩ :=
      iteration limit p bytes entered heap address lower (Nat.le_sqrt.mp inside)
        (by simp [entered, setLocal]) enteredN enteredArray found sieved
    obtain ⟨finalEnv, finalBytes, next, executed, finalSieved, finished, saved⟩ :=
      loop limit (p+1) middleBytes middleEnv (heap.set address (.bytearray middleBytes)) address
        (by omega) (stable.1.trans enteredN) (stable.2.trans enteredArray)
        (List.getElem?_set_self (heap_inside found)) nextSieved
    refine ⟨finalEnv, finalBytes, next, ?_, finalSieved, finished, ?_, ?_⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, loopBody, List.set_set] using
        iterated.append executed
    · simpa [entered, setLocal] using saved.1.trans stable.1
    · simpa [entered, setLocal] using saved.2.trans stable.2
  · refine ⟨env, bytes, p, ?_, sieved, Nat.sqrt_lt'.mp (by omega), rfl, rfl⟩
    rw [heap_set_same heap address _ found]
    exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
termination_by Nat.sqrt limit + 1 - p
decreasing_by omega

private def primeBody (collector : Nat) : List Stmt :=
  [.ifThen (.subscript (.name "is_prime") (.name "i")) [.comprehensionAppend collector (.name "i")] []]

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
    (correct : ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i))
    (items : List Nat) (env : HeapCalls.Env) (getArray : env "is_prime" = some (.ref address)) (lower : 2 ≤ first) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "i" first stop 1 (primeBody base.length) []]
      (base ++ [.list (encode items)]) finalEnv
      (base ++ [.list (encode (items ++ (List.range' first count).filter fun p => decide (Nat.Prime p)))]) := by
  induction count generalizing first items env with
  | zero =>
      refine ⟨env, ?_⟩
      simp only [List.range'_zero, List.filter_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "i" (number first)
      have getArray' : bound "is_prime" = some (.ref address) := by simpa [bound, setLocal] using getArray
      have inside : first < bytes.length := by omega
      have tested := read_prime (base ++ [.list (encode items)]) bound address first bytes inside
        (by simp [bound, setLocal]) getArray' (found_append found _)
      have live := correct first bytes[first] lower (List.getElem?_eq_getElem inside)
      by_cases prime : Nat.Prime first
      · have emitted : BlockContinues program bound (primeBody base.length)
            (base ++ [.list (encode items)]) bound (base ++ [.list (encode (items ++ [first]))]) := by
          apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, live.mpr prime])
          have write := BlockContinues.comprehension_append (globals := []) (module := program) (env := bound)
            (address := base.length) (heap := base ++ [.list (encode items)])
            (Evaluates.lookup (name := "i") (value := number first) (by simp [bound, setLocal]))
            (by simp : (base ++ [.list (encode items)])[base.length]? = some (.list (encode items)))
          simpa [encode, List.map_append] using write
        obtain ⟨finalEnv, rest⟩ := ih (first + 1) (by omega) (items ++ [first]) bound getArray' (by omega)
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, primeBody,
          Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.filter_cons,
          show decide (Nat.Prime first) = true by simp [prime], if_true, Int.natCast_add,
          Int.natCast_one, List.append_assoc, List.singleton_append] using emitted.append rest
      · have skipped : BlockContinues program bound (primeBody base.length)
            (base ++ [.list (encode items)]) bound (base ++ [.list (encode items)]) := by
          have zero : bytes[first].toNat = 0 := by tauto
          exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, zero]) .nil
        obtain ⟨finalEnv, rest⟩ := ih (first + 1) (by omega) items bound getArray' (by omega)
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, primeBody,
          Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.filter_cons,
          show decide (Nat.Prime first) = false by simp [prime], Int.natCast_add,
          Int.natCast_one] using skipped.append rest


private theorem prime_range (limit : Nat) (large : 2 ≤ limit) :
    (List.range' 2 (limit-1)).filter Nat.Prime = ProjectEulerStatements.PrimePartitions.primesUpTo limit := by
  have split : List.range (limit+1) = [0,1] ++ List.range' 2 (limit-1) := by
    rw [List.range_eq_range', show limit+1 = 2+(limit-1) by omega, ← List.range'_append_1]
    rfl
  simp [ProjectEulerStatements.PrimePartitions.primesUpTo, split, Nat.not_prime_zero, Nat.not_prime_one]

private theorem collected (limit : Nat) (large : 2 ≤ limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (bytes : List UInt8) (size : bytes.length = limit+1)
    (getLimit : env "n" = some (number limit)) (getArray : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes))
    (correct : ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Evaluates program env
      (.listComp (.name "i") [("i", .call (.name "range")
        [.constant (.int 2), .binOp (.name "n") .add (.constant (.int 1))],
        [.subscript (.name "is_prime") (.name "i")])]) heap (.ref heap.length)
      (heap ++ [.list (encode (ProjectEulerStatements.PrimePartitions.primesUpTo limit))]) := by
  let inner := comprehensionEnv env [("i", .call (.name "range")
    [.constant (.int 2), .binOp (.name "n") .add (.constant (.int 1))],
    [.subscript (.name "is_prime") (.name "i")])]
  obtain ⟨finalEnv, scanned⟩ := prime_loop 2 (limit-1) (limit+1) (by omega) heap address bytes size found correct [] inner
    (by simpa [inner, comprehensionEnv] using getArray) (by omega)
  apply Evaluates.list_comp_range (finalEnv := finalEnv) (values := [number 2, number (limit+1)])
    (.cons Evaluates.constant (.cons
      (Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
  simpa only [encode, List.map_nil, List.nil_append, prime_range limit large] using scanned

/-- The source returns the entire inclusive prime list while preserving every caller object. -/
theorem positive_result (limit : Nat) (large : 2 ≤ limit) (heap : Mutable.Heap) :
    ∃ bytes, ReturnsValue program "sieve_primes" [number limit] heap (.ref (heap.length+2))
      (heap ++ [.bytearray [1], .bytearray bytes,
        .list (encode (ProjectEulerStatements.PrimePartitions.primesUpTo limit))]) := by
  let args := ((findFunctionIn program.body "sieve_primes").getD ([], [])).1
  let body := ((findFunctionIn program.body "sieve_primes").getD ([], [])).2
  let env := functionEnv args body [number limit] []
  let ready := setLocal (setLocal env "is_prime" (.ref (heap.length+1))) "limit" (number (Nat.sqrt limit))
  have getLimit : ready "n" = some (number limit) := rfl
  have getArray : ready "is_prime" = some (.ref (heap.length+1)) := rfl
  obtain ⟨finalEnv, bytes, next, executed, sieved, finished, stable⟩ :=
    loop limit 2 (initial limit) ready (initialHeap heap limit) (heap.length+1) (by omega) getLimit getArray
      (by simp [initialHeap]) (sieved_initial limit large)
  let middle := heap ++ [.bytearray [1], .bytearray bytes]
  have loopRun : BlockContinues program ready [code[4]'(by decide)] (initialHeap heap limit) finalEnv middle := by
    apply BlockContinues.for_range (middle := initialHeap heap limit)
      (values := [number 2, number (Nat.sqrt limit + 1)])
    · exact .cons Evaluates.constant (.cons
        (Evaluates.binary (middle := initialHeap heap limit) (afterRight := initialHeap heap limit)
          (av := number (Nat.sqrt limit)) (bv := number 1)
          (Evaluates.lookup (show ready "limit" = some (number (Nat.sqrt limit)) from rfl))
          Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil)
    · rfl
    · simpa [initialHeap, middle] using executed
  have correct : ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i) := by
    intro i byte lower found
    have inside := (List.getElem?_eq_some_iff.mp found).1
    exact (sieved.2 i byte lower found).trans
      (survives_finished limit next i lower (by have := sieved.1; omega) finished)
  refine ⟨bytes, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) (middle := heap) (value := .scalar (.bool false))
  · apply Evaluates.compare (Evaluates.lookup (show env "n" = some (number limit) from rfl)) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  · rfl
  · apply BlockContinues.returns ((setup limit large env heap rfl).append loopRun)
    apply BlockReturns.return_value
    have read := collected limit large middle finalEnv (heap.length+1) bytes sieved.1
      (stable.1.trans getLimit) (stable.2.trans getArray) (by simp [middle]) correct
    simpa [middle, List.append_assoc] using read

/-- Small signed limits produce an empty list; larger limits run the complete sieve. -/
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ address extra, ReturnsValue program "sieve_primes" [integer limit] heap (.ref address) (heap ++ extra) ∧
      (heap ++ extra)[address]? = some (.list (encode (ProjectEulerStatements.PrimePartitions.primesUpTo limit.toNat))) := by
  by_cases small : limit < 2
  · have empty : ProjectEulerStatements.PrimePartitions.primesUpTo limit.toNat = [] := by
      have bound : limit.toNat ≤ 1 := by omega
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro p member
      have inside : p < limit.toNat+1 := List.mem_range.mp (List.mem_filter.mp member).1
      have prime : Nat.Prime p := of_decide_eq_true (List.mem_filter.mp member).2
      have := prime.two_le
      omega
    refine ⟨heap.length, [.list []], ?_, by simp [empty, encode]⟩
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := true) (middle := heap) (value := .scalar (.bool true))
    · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, small]
    · rfl
    · exact BlockReturns.return_value (Evaluates.list .nil)
  · lift limit to Nat using (by omega : 0 ≤ limit)
    obtain ⟨bytes, executed⟩ := positive_result limit (by omega) heap
    exact ⟨heap.length+2, [.bytearray [1], .bytearray bytes,
      .list (encode (ProjectEulerStatements.PrimePartitions.primesUpTo limit))], executed, by simp⟩

end sieve_primes
end solvers.«87»
