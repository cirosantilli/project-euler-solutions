import «.formalsw».solvers.«51.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.LoopCalls
import ProjectEulerSolutions.Lib.PrimeSieve
import ProjectEulerSolutions.Lib.PrimeReplacement
import ProjectEulerSolutions.Lib.BitSubsets
import ProjectEulerSolutions.Lib.HeapCollection
import Mathlib.Data.Nat.Sqrt

/-! Source proofs for problem 51's sieve, replacement counts, and bit-mask selection. -/

set_option Elab.async false

namespace solvers.«51»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P51Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

namespace sieve
open ProjectEulerSolutions.PrimeSieve
private def code : List Stmt := executableBody ((findFunctionIn program.body "sieve").getD ([], [])).2
private def loopBody : List Stmt := match (code[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def markBody : List Stmt := match (loopBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body | _ => []
private def markEnv (env : HeapCalls.Env) (p : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "start" (number (p*p : Nat))) "step" (number p)

private theorem mark_result (env : HeapCalls.Env) (heap : Mutable.Heap) (address limit p : Nat)
    (bytes : List UInt8) (lower : 2 ≤ p) (inside : p*p ≤ limit) (size : bytes.length = limit+1)
    (getP : env "i" = some (number p)) (getLimit : env "n" = some (number limit))
    (getBuffer : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv, BlockContinues program env markBody heap finalEnv (heap.set address (.bytearray (mark bytes p))) ∧
      finalEnv "n" = env "n" ∧ finalEnv "is_prime" = env "is_prime" := by
  let e1 := setLocal env "start" (number (p*p : Nat))
  let e2 := markEnv env p
  let count : Int := ((limit : Int)-p*p)/p+1
  have first : BlockContinues program env [markBody[0]'(by decide)] heap e1 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP)
      (by simp [Mutable.binary, scalarBinary])
  have second : BlockContinues program e1 [markBody[1]'(by decide)] heap e2 heap :=
    BlockContinues.assign (Evaluates.lookup (by simpa [e1, setLocal] using getP))
  have third : BlockContinues program e2 [markBody[2]'(by decide)] heap e2
      (heap.set address (.bytearray (mark bytes p))) := by
    apply BlockContinues.setSlice (value := .scalar (.bytes (List.replicate count.toNat 0)))
      (afterRhs := heap) (afterTarget := heap) (receiver := .ref address)
      (lowerValue := number (p*p : Nat)) (upperValue := number (limit+1 : Nat)) (strideValue := number p)
    · apply Evaluates.binary (av := .scalar (.bytes [0])) (bv := number count) (middle := heap) (afterRight := heap)
      · exact Evaluates.constant
      · apply Evaluates.binary (av := number (((limit : Int)-p*p)/p)) (bv := number 1)
          (middle := heap) (afterRight := heap)
        · apply Evaluates.binary (av := number ((limit : Int)-p*p)) (bv := number p)
            (middle := heap) (afterRight := heap)
          · exact Evaluates.binary (av := number limit) (bv := number (p*p : Nat))
              (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getLimit))
              (Evaluates.lookup (by simp [e2, markEnv, setLocal])) (by simp [Mutable.binary, scalarBinary, number])
          · exact Evaluates.lookup (by simp [e2, markEnv, setLocal])
          · simp [Mutable.binary, scalarBinary, number, show p ≠ 0 by omega, show 0 < p by omega]
        · exact Evaluates.constant
        · rfl
      · simp [Mutable.binary, scalarBinary]
    · exact .cons (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getBuffer))
        (.cons (Evaluates.lookup (by simp [e2, markEnv, setLocal]))
          (.cons (Evaluates.binary (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getLimit))
            Evaluates.constant (by simp [Mutable.binary, scalarBinary, number]))
          (.cons (Evaluates.lookup (by simp [e2, markEnv, setLocal])) .nil)))
    · apply Mutable.setSlice_buffer heap address bytes _ _ _ _ _
        { start := some (p*p), stop := some (limit+1), step := some p } found
      · change ((if some (p : Int) = some 0 then Mutable.error "ValueError"
          else (pure { start := some ((p : Int)*p), stop := some ((limit : Int)+1), step := some (p : Int) } : Mutable.Eval FormalSW.Py.Builtin.Slice.Slice)) heap) = _
        rw [if_neg (by simp; omega)]
        simp
      · have result := mark_slice bytes p lower (by omega)
        have equal : (bytes.length : Int)-p*p-1 = (limit : Int)-p*p := by rw [size]; push_cast; ring
        rw [equal] at result
        simpa only [size, count, Nat.cast_add, Nat.cast_one] using result
  exact ⟨e2, first.append (second.append third), by simp [e2, markEnv, setLocal], by simp [e2, markEnv, setLocal]⟩

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
  after "n" = before "n" ∧ after "is_prime" = before "is_prime"

private theorem read_buffer (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat)
    (bytes : List UInt8) (valid : p < bytes.length) (getP : env "i" = some (number p))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    Evaluates program env (.subscript (.name "is_prime") (.name "i")) heap (number bytes[p].toNat) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getArray) (Evaluates.lookup getP)
  exact Mutable.getItem_buffer heap address p bytes found valid

private theorem iteration (limit p : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (large : 1 ≤ limit) (lower : 2 ≤ p) (inside : p * p ≤ limit)
    (getP : env "i" = some (number p)) (getLimit : env "n" = some (number limit))
    (getArray : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes))
    (invariant : Invariant limit p bytes) :
    ∃ finalEnv finalBytes, BlockContinues program env loopBody heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Invariant limit (p + 1) finalBytes ∧ Stable env finalEnv := by
  have size := invariant.1.1
  have valid : p < bytes.length := by nlinarith
  have live : bytes[p].toNat ≠ 0 ↔ Nat.Prime p :=
    (invariant.1.2 p bytes[p] lower (List.getElem?_eq_getElem valid)).trans (survives_current p lower)
  by_cases prime : Nat.Prime p
  · obtain ⟨finalEnv, marked, savedLimit, savedArray⟩ := mark_result env heap address limit p bytes
      lower inside size getP getLimit getArray found
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
    (getLimit : env "n" = some (number limit)) (getArray : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (invariant : Invariant limit first bytes) :
    ∃ finalEnv finalBytes, BlockContinues program env
      [.rangeStepLoop "i" first (Nat.sqrt limit + 1) 1 loopBody []] heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Invariant limit (Nat.sqrt limit + 1) finalBytes ∧ Stable env finalEnv := by
  induction count generalizing first bytes env heap with
  | zero =>
      have same : first = Nat.sqrt limit + 1 := by omega
      refine ⟨env, bytes, ?_, by simpa only [same] using invariant, rfl, rfl⟩
      rw [heap_set_same heap address _ found]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "i" (number first)
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


private theorem initial_slice (limit : Nat) (large : 1 ≤ limit) :
    FormalSW.Py.Builtin.ByteArray.setSlice (List.replicate (limit+1) 1)
      { start := some 0, stop := some 2 } [0,0] = .ok (initial limit) := by
  have normalized : FormalSW.Py.Builtin.Slice.indices
      { start := some 0, stop := some 2 } (limit+1) = .ok (0,2,1) := by
    simp only [FormalSW.Py.Builtin.Slice.indices, Option.getD_none]
    change Except.ok (max (0 : Int) (min 0 ((limit+1 : Nat) : Int)), max 0 (min 2 ((limit+1 : Nat) : Int)), (1 : Int)) = _
    simp
    omega
  simp only [FormalSW.Py.Builtin.ByteArray.setSlice, FormalSW.Py.Builtin.List.setSlice,
    List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte, List.length_replicate, normalized]
  change Except.ok ([0,0] ++ (List.replicate (limit+1) 1).drop 2) = _
  simp [initial, show limit+1-2 = limit-1 by omega]

private def initializedEnv (env : HeapCalls.Env) (address limit : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "is_prime" (.ref address)) "r" (number (Nat.sqrt limit))
private def initialHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray (initial limit)]
private theorem setup (limit : Nat) (large : 1 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "n" = some (number limit)) :
    BlockContinues program env ((code.drop 1).take 3) heap
      (initializedEnv env (heap.length+1) limit) (initialHeap heap limit) := by
  let e1 := setLocal env "is_prime" (.ref (heap.length+1))
  let allocated := heap ++ [.bytearray [1], .bytearray (List.replicate (limit+1) 1)]
  have first : BlockContinues program env [code[1]'(by decide)] heap e1 allocated := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := .ref heap.length) (middle := heap ++ [.bytearray [1]])
      (bv := number (limit+1 : Nat)) (afterRight := heap ++ [.bytearray [1]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
    · simp [allocated, Mutable.binary, Mutable.allocateBuffer, Except.map,
        show (BinOp.mult == .mult) = true from rfl]
  have second : BlockContinues program e1 [code[2]'(by decide)] allocated e1 (initialHeap heap limit) := by
    have found : allocated[heap.length+1]? = some (.bytearray (List.replicate (limit+1) 1)) := by simp [allocated]
    have write := Mutable.setSlice_buffer allocated (heap.length+1) (List.replicate (limit+1) 1) [0,0] (initial limit)
      (.scalar (.int 0)) (.scalar (.int 2)) (.scalar .none)
      { start := some 0, stop := some 2 } found rfl (initial_slice limit large)
    have same : allocated.set (heap.length+1) (.bytearray (initial limit)) = initialHeap heap limit := by
      simp [allocated, initialHeap]
    rw [same] at write
    exact BlockContinues.setSlice Evaluates.constant
      (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))) write
  have third : BlockContinues program e1 [code[3]'(by decide)] (initialHeap heap limit)
      (initializedEnv env (heap.length+1) limit) (initialHeap heap limit) := by
    apply BlockContinues.assign
    apply Evaluates.library_call (.cons (Evaluates.lookup (by simpa [e1, setLocal] using getLimit)) .nil) rfl
    change scalarResult (some (Math.isqrtCall [.int limit])) (initialHeap heap limit) = _
    have root : Math.isqrt limit = Nat.sqrt limit := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
    simp [scalarResult, Math.isqrtCall, root, show ¬ (limit : Int) < 0 by omega]
  exact first.append (second.append third)


/-- The exact source prefix initializes and marks an inclusive primality table, preserving the input heap. -/
theorem prepare (limit : Nat) (large : 1 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "n" = some (number limit)) :
    ∃ finalEnv bytes, BlockContinues program env ((code.drop 1).take 4) heap finalEnv
        (heap ++ [.bytearray [1], .bytearray bytes]) ∧
      finalEnv "n" = some (number limit) ∧ finalEnv "is_prime" = some (.ref (heap.length + 1)) ∧
      bytes.length = limit + 1 ∧ (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  let ready := initializedEnv env (heap.length + 1) limit
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
  have readyLimit : ready "n" = some (number limit) := by simpa [ready, initializedEnv, setLocal] using getLimit
  have rootPositive : 1 ≤ Nat.sqrt limit := by simpa using Nat.sqrt_le_sqrt large
  obtain ⟨finalEnv, bytes, executed, sieved, stable⟩ := loop limit 2 (Nat.sqrt limit - 1) (initial limit)
    ready (initialHeap heap limit) (heap.length + 1) large (by omega) (by omega) readyLimit
    (by simp [ready, initializedEnv, setLocal]) (by simp [initialHeap]) initialInvariant
  have loopRun : BlockContinues program ready [code[4]'(by decide)] (initialHeap heap limit) finalEnv
      (heap ++ [.bytearray [1], .bytearray bytes]) := by
    apply BlockContinues.for_range (middle := initialHeap heap limit)
      (values := [number 2, number (Nat.sqrt limit + 1 : Nat)])
      (.cons Evaluates.constant (.cons (Evaluates.binary (av := number (Nat.sqrt limit)) (bv := number 1)
        (middle := initialHeap heap limit) (afterRight := initialHeap heap limit)
        (Evaluates.lookup (by simp [ready, initializedEnv, setLocal])) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) .nil)) rfl
    simpa [initialHeap] using executed
  refine ⟨finalEnv, bytes, (setup limit large env heap getLimit).append loopRun,
    stable.1.trans readyLimit, stable.2.trans (by simp [ready, initializedEnv, setLocal]), sieved.1.1, ?_⟩
  intro i byte found
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

/-- Increasing primes in the requested inclusive interval. -/
def primes (limit : Nat) : List Nat := (List.range' 2 (limit-1)).filter Nat.Prime
def numbers (values : List Nat) : List Mutable.Value := values.map (fun (n : Nat) => number n)
private def selected (start count : Nat) : List Nat := (List.range' start count).filter Nat.Prime
private abbrev collectBody (collector : Nat) : List Stmt :=
  [.ifThen (.subscript (.name "is_prime") (.name "i")) [.comprehensionAppend collector (.name "i")] []]
private theorem collect_loop (count start limit : Nat) (length : start+count = limit+1) (lower : 2 ≤ start)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (address collector : Nat) (separate : address ≠ collector)
    (bytes : List UInt8) (before : List Nat)
    (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = limit+1)
    (correct : ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i))
    (collected : heap[collector]? = some (.list (numbers before))) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "i" start (limit+1) 1 (collectBody collector) []] heap finalEnv
      (heap.set collector (.list (numbers (before ++ selected start count)))) := by
  induction count generalizing start heap env before with
  | zero =>
      have done : start = limit+1 := by omega
      refine ⟨env, ?_⟩
      simp only [selected, List.range'_zero, List.filter_nil, List.append_nil, heap_set_same heap collector _ collected]
      apply BlockContinues.range_done _ .nil
      simp [Calls.rangeContinues, done]
  | succ count ih =>
      have valid : start < bytes.length := by omega
      let bound := setLocal env "i" (number start)
      have getBuffer' : bound "is_prime" = some (.ref address) := by simpa [bound, setLocal] using getBuffer
      have read : Evaluates program bound (.subscript (.name "is_prime") (.name "i")) heap
          (number bytes[start].toNat) heap := by
        apply Evaluates.subscript (middle := heap) (afterIndex := heap) (by intros; intro h; cases h)
          (Evaluates.lookup getBuffer') (Evaluates.lookup rfl)
        exact Mutable.getItem_buffer heap address start bytes found valid
      have live := correct start bytes[start] lower (List.getElem?_eq_getElem valid)
      by_cases prime : Nat.Prime start
      · let middle := heap.set collector (.list (numbers (before ++ [start])))
        have middleFound : middle[address]? = some (.bytearray bytes) := by
          simpa [middle, List.getElem?_set_ne (Ne.symm separate)] using found
        have middleCollected : middle[collector]? = some (.list (numbers (before ++ [start]))) :=
          List.getElem?_set_self (heap_inside collected)
        have emitted : BlockContinues program bound (collectBody collector) heap bound middle := by
          apply BlockContinues.branch (flag := true) read
          · simp [Mutable.truth, scalarTruth, live.mpr prime]
          · simpa [middle, numbers] using BlockContinues.comprehension_append
              (Evaluates.lookup (globals := []) (module := program) (heap := heap) (show bound "i" = some (number start) from rfl)) collected
        obtain ⟨finalEnv, executed⟩ := ih (start+1) (by omega) (by omega) middle bound (before ++ [start])
          getBuffer' middleFound middleCollected
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, collectBody,
          middle, List.set_set, selected, List.range'_succ, prime, List.append_assoc, bound, Nat.cast_add, Nat.cast_one]
          using emitted.append executed
      · have skipped : BlockContinues program bound (collectBody collector) heap bound heap := by
          apply BlockContinues.branch (flag := false) read
          · have zero : bytes[start].toNat = 0 := by tauto
            simp [Mutable.truth, scalarTruth, zero]
          · exact .nil
        obtain ⟨finalEnv, executed⟩ := ih (start+1) (by omega) (by omega) heap bound before getBuffer' found collected
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, collectBody,
          selected, List.range'_succ, prime, bound, Nat.cast_add, Nat.cast_one] using skipped.append executed

private abbrev outputExpression : Expr :=
  .listComp (.name "i") [("i", .call (.name "range") [.constant (.int 2), .binOp (.name "n") .add (.constant (.int 1))],
    [.subscript (.name "is_prime") (.name "i")])]
private theorem collect (limit : Nat) (large : 1 ≤ limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (bytes : List UInt8)
    (getLimit : env "n" = some (number limit)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = limit+1)
    (correct : ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Evaluates program env outputExpression heap (.ref heap.length) (heap ++ [.list (numbers (primes limit))]) := by
  have bound := heap_inside found
  obtain ⟨finalEnv, collected⟩ := collect_loop (limit-1) 2 limit (by omega) (by omega)
    (heap ++ [.list []])
    (comprehensionEnv env [("i", .call (.name "range") [.constant (.int 2), .binOp (.name "n") .add (.constant (.int 1))],
      [.subscript (.name "is_prime") (.name "i")])]) address heap.length (by omega) bytes []
    (by simpa [comprehensionEnv] using getBuffer)
    (by rw [List.getElem?_append_left bound]; exact found) size correct (by simp [numbers])
  apply Evaluates.list_comp_range (values := [number 2, number (limit+1 : Nat)]) (middle := heap)
  · exact .cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup getLimit)
      Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])) .nil)
  · rfl
  · simpa [primes, selected, List.set_append, comprehensionFilters, comprehensionBody, collectBody] using collected



/-- The source returns the sieve and its increasing prime list in a fresh tuple. -/
theorem positive_result (limit : Nat) (large : 1 ≤ limit) (heap : Mutable.Heap) :
    ∃ bytes, ReturnsValue program "sieve" [number limit] heap (.ref (heap.length + 3))
        (heap ++ [.bytearray [1], .bytearray bytes, .list (numbers (primes limit)),
          .tuple [.ref (heap.length + 1), .ref (heap.length + 2)]]) ∧
      bytes.length = limit + 1 ∧ (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  let env := functionEnv ((findFunctionIn program.body "sieve").getD ([], [])).1
    ((findFunctionIn program.body "sieve").getD ([], [])).2 [number limit] []
  obtain ⟨finalEnv, bytes, prepared, getLimit, getBuffer, size, correct⟩ := prepare limit large env heap rfl
  let middle := heap ++ [Mutable.Object.bytearray [1], .bytearray bytes]
  have found : middle[heap.length + 1]? = some (.bytearray bytes) := by simp [middle]
  have collected := collect limit large middle finalEnv (heap.length + 1) bytes getLimit getBuffer found size
    (fun i byte _ found => correct i byte found)
  let afterList := middle ++ [.list (numbers (primes limit))]
  let ready := setLocal finalEnv "primes" (.ref middle.length)
  have returned : Evaluates program ready (.tuple [.name "is_prime", .name "primes"]) afterList (.ref afterList.length)
      (afterList ++ [.tuple [.ref (heap.length + 1), .ref middle.length]]) := by
    apply Evaluates.tuple (.cons (Evaluates.lookup (by simpa [ready, setLocal] using getBuffer)) (.cons (Evaluates.lookup rfl) .nil))
    simp
  refine ⟨bytes, ?_, size, correct⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) (value := flag false) (middle := heap)
  · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
  · rfl
  · have finished := prepared.returns (BlockReturns.assign (name := "primes") collected (BlockReturns.return_value (rest := []) returned))
    simpa only [afterList, middle, List.length_append, List.length_cons, List.length_nil, Nat.add_zero,
      List.append_assoc, List.singleton_append, Nat.add_assoc] using finished


/-- The early source branch returns an empty prime list and a correctly sized zero buffer. -/
theorem nonpositive_result (limit : Int) (small : limit < 1) (heap : Mutable.Heap) :
    ReturnsValue program "sieve" [number limit] heap (.ref (heap.length + 3))
      (heap ++ [.bytearray [0], .bytearray (List.replicate (limit + 1).toNat 0), .list [],
        .tuple [.ref (heap.length + 1), .ref (heap.length + 2)]]) := by
  let env := functionEnv ((findFunctionIn program.body "sieve").getD ([], [])).1
    ((findFunctionIn program.body "sieve").getD ([], [])).2 [number limit] []
  let bufferExpression : Expr := .binOp (.builtinCall .bytearray [.constant (.bytes [0])]) .mult
    (.binOp (.name "n") .add (.constant (.int 1)))
  let afterBuffer := heap ++ [.bytearray [0], .bytearray (List.replicate (limit + 1).toNat 0)]
  let afterList := afterBuffer ++ [.list []]
  have buffer : Evaluates program env bufferExpression heap (.ref (heap.length + 1)) afterBuffer := by
    apply Evaluates.binary (av := .ref heap.length) (bv := number (limit + 1))
      (middle := heap ++ [.bytearray [0]]) (afterRight := heap ++ [.bytearray [0]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.binary (av := number limit) (bv := number 1) (Evaluates.lookup rfl) Evaluates.constant rfl
    · simp [afterBuffer, Mutable.binary, Mutable.allocateBuffer, Except.map, show (BinOp.mult == .mult) = true from rfl]
  have empty : Evaluates program env (.list []) afterBuffer (.ref afterBuffer.length) afterList := Evaluates.list .nil
  have pair : Evaluates program env (.tuple [bufferExpression, .list []]) heap (.ref afterList.length)
      (afterList ++ [.tuple [.ref (heap.length + 1), .ref afterBuffer.length]]) :=
    Evaluates.tuple (.cons buffer (.cons empty .nil)) (by simp)
  have returned : ReturnsValue program "sieve" [number limit] heap (.ref afterList.length)
      (afterList ++ [.tuple [.ref (heap.length + 1), .ref afterBuffer.length]]) := by
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := true) (middle := heap) (value := flag true)
    · exact Evaluates.compare (av := number limit) (bv := number 1) (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])
    · rfl
    · exact BlockReturns.return_value pair
  simpa only [afterList, afterBuffer, List.length_append, List.length_cons, List.length_nil, Nat.add_zero,
    List.append_assoc, List.singleton_append, Nat.add_assoc] using returned

/-- Every signed bound returns an inclusive primality buffer and the increasing list of its primes. -/
theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ seed bytes, ReturnsValue program "sieve" [number limit] heap (.ref (heap.length + 3))
        (heap ++ [.bytearray seed, .bytearray bytes, .list (numbers (primes limit.toNat)),
          .tuple [.ref (heap.length + 1), .ref (heap.length + 2)]]) ∧
      bytes.length = (limit + 1).toNat ∧ (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  by_cases small : limit < 1
  · have zero : limit.toNat = 0 := by omega
    refine ⟨[0], List.replicate (limit + 1).toNat 0, ?_, by simp, ?_⟩
    · simpa only [zero, primes, Nat.zero_sub, List.range'_zero, List.filter_nil, numbers, List.map_nil] using nonpositive_result limit small heap
    · intro i byte found
      have inside := (List.getElem?_eq_some_iff.mp found).1
      simp only [List.length_replicate] at inside
      have same : i = 0 := by omega
      have zero : byte = 0 := List.eq_of_mem_replicate (List.mem_of_getElem? found)
      simp [same, zero, Nat.not_prime_zero]
  · lift limit to Nat using (by omega : 0 ≤ limit)
    obtain ⟨bytes, returned, size, correct⟩ := positive_result limit (by omega) heap
    exact ⟨[1], bytes, returned, by simpa using size, correct⟩

theorem mem_primes (limit p : Nat) : p ∈ primes limit ↔ Nat.Prime p ∧ p ≤ limit := by
  simp only [primes, List.mem_filter, List.mem_range', decide_eq_true_eq]
  constructor
  · rintro ⟨⟨i, bound, rfl⟩, prime⟩; exact ⟨prime, by omega⟩
  · rintro ⟨prime, upper⟩; exact ⟨⟨p-2, by have := prime.two_le; omega, by have := prime.two_le; omega⟩, prime⟩

theorem primes_pairwise (limit : Nat) : (primes limit).Pairwise (· < ·) :=
  (List.pairwise_lt_range' (s := 2) (n := limit-1)).filter _

end sieve

private def replacementExpression (originalName : String) : Expr :=
  .binOp (.name "p") .add (.binOp (.binOp (.name "r") .sub (.name originalName)) .mult (.name "shift"))

private theorem replacement_evaluates (originalName : String) (p : Nat) (indices : List Nat) (original digit : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getP : env "p" = some (number p))
    (getDigit : env "r" = some (number digit)) (getOriginal : env originalName = some (number original))
    (getShift : env "shift" = some (number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat))) :
    Evaluates program env (replacementExpression originalName) heap
      (number (Nat.ofDigits 10 (ProjectEulerStatements.P51.replaceAt (Nat.digits 10 p)
        (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices) digit) : Nat)) heap := by
  have calculated : Evaluates program env (replacementExpression originalName) heap
      (number ((p : Int) + ((digit : Int) - original) *
        ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat))) heap := by
    apply Evaluates.binary (av := number p) (bv := number (((digit : Int) - original) *
      ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat)))
      (middle := heap) (afterRight := heap)
      (Evaluates.lookup getP)
    · apply Evaluates.binary (av := number ((digit : Int) - original))
        (bv := number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat))
        (middle := heap) (afterRight := heap)
        (Evaluates.binary (av := number digit) (bv := number original) (Evaluates.lookup getDigit) (Evaluates.lookup getOriginal)
          (by simp [Mutable.binary, scalarBinary]))
        (Evaluates.lookup getShift)
      simp [Mutable.binary, scalarBinary]
    · simp [Mutable.binary, scalarBinary]
  have equal := ProjectEulerSolutions.PrimeReplacement.replacement_value_reflected (Nat.digits 10 p) indices original digit distinct bounded same
  rw [Nat.ofDigits_digits] at equal
  rw [← equal] at calculated
  exact calculated

namespace replacementCounter
private def originalName (inSearch : Bool) : String := if inSearch then "d" else "d0"
private def loopBody (inSearch : Bool) : List Stmt :=
  [.ifThen (.boolOp .and [.name "lead", .compare (.name "r") [(.eq, .constant (.int 0))]]) [.continueLoop] [],
   .assign [.name "num"] (replacementExpression (originalName inSearch)),
   .ifThen (.subscript (.name "is_prime") (.name "num")) [.augAssign (.name "cnt") .add (.constant (.int 1))] []]

/-- The shared replacement assignment computes the reference replacement at reflected digit positions. -/
theorem replacement_assignment (inSearch : Bool) (p : Nat) (indices : List Nat) (original digit : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getP : env "p" = some (number p))
    (getDigit : env "r" = some (number digit)) (getOriginal : env (originalName inSearch) = some (number original))
    (getShift : env "shift" = some (number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat))) :
    BlockContinues program env [(loopBody inSearch)[1]'(by cases inSearch <;> decide)] heap
      (setLocal env "num" (number (Nat.ofDigits 10 (ProjectEulerStatements.P51.replaceAt (Nat.digits 10 p)
        (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices) digit) : Nat))) heap := by
  apply BlockContinues.assign
  exact replacement_evaluates (originalName inSearch) p indices original digit distinct bounded same heap env getP getDigit getOriginal getShift

private abbrev candidate (p : Nat) (indices : List Nat) (digit : Nat) : Nat :=
  Nat.ofDigits 10 (ProjectEulerStatements.P51.replaceAt (Nat.digits 10 p)
    (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices) digit)

private def accepted (p : Nat) (indices : List Nat) (digit : Nat) : Bool :=
  decide (digit ≠ 0 ∨ 0 ∉ indices) && ProjectEulerStatements.P51.isPrime (candidate p indices digit)

private structure Inputs (inSearch : Bool) (p : Nat) (indices : List Nat) (original address : Nat) (env : HeapCalls.Env) : Prop where
  getP : env "p" = some (number p)
  getOriginal : env (originalName inSearch) = some (number original)
  getShift : env "shift" = some (number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat))
  getLead : env "lead" = some (flag (decide (0 ∈ indices)))
  getBuffer : env "is_prime" = some (.ref address)

private theorem inputs_set (inSearch : Bool) (p : Nat) (indices : List Nat) (original address : Nat)
    (env : HeapCalls.Env) (inputs : Inputs inSearch p indices original address env) (name : String) (value : Mutable.Value)
    (different : name ≠ "p" ∧ name ≠ originalName inSearch ∧ name ≠ "shift" ∧ name ≠ "lead" ∧ name ≠ "is_prime") :
    Inputs inSearch p indices original address (setLocal env name value) := by
  rcases inputs with ⟨hp, hd, hs, hl, hb⟩
  exact ⟨by simpa [setLocal, Ne.symm different.1] using hp,
    by simpa [setLocal, Ne.symm different.2.1] using hd,
    by simpa [setLocal, Ne.symm different.2.2.1] using hs,
    by simpa [setLocal, Ne.symm different.2.2.2.1] using hl,
    by simpa [setLocal, Ne.symm different.2.2.2.2] using hb⟩

private theorem leading_test (inSearch : Bool) (p : Nat) (indices : List Nat) (original address digit : Nat)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (inputs : Inputs inSearch p indices original address env)
    (getDigit : env "r" = some (number digit)) :
    Evaluates program env (.boolOp .and [.name "lead", .compare (.name "r") [(.eq, .constant (.int 0))]])
      heap (flag (decide (0 ∈ indices ∧ digit = 0))) heap := by
  by_cases leading : 0 ∈ indices
  · apply Evaluates.boolean_next (value := flag true) (flag := true)
      (by simpa [leading] using Evaluates.lookup (module := program) (heap := heap) inputs.getLead)
      (by simp [Mutable.truth, scalarTruth]) (by cases inSearch <;> decide)
    apply Evaluates.boolean_last (flag := decide (digit = 0))
    · apply Evaluates.compare (av := number digit) (bv := number 0)
        (middle := heap) (afterRight := heap) (Evaluates.lookup getDigit) Evaluates.constant
      simp [HeapCalls.compare, compareScalars, scalarCompare, evalCmpOp, leading, flag]
    · simp [Mutable.truth, scalarTruth, leading]
  · simpa [leading] using Evaluates.boolean_stop (op := BoolOp.and)
      (rest := [.compare (.name "r") [(.eq, .constant (.int 0))]])
      (by simpa [leading] using Evaluates.lookup (module := program) (heap := heap) inputs.getLead)
      (flag := false) (by simp [Mutable.truth, scalarTruth]) (by cases inSearch <;> decide)

private def Stable (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "r" → name ≠ "num" → name ≠ "cnt" → after name = before name

private theorem stable_refl (env : HeapCalls.Env) : Stable env env := fun _ _ _ _ => rfl

private theorem stable_trans (first : Stable before middle) (second : Stable middle after) : Stable before after :=
  fun name hr hn hc => (second name hr hn hc).trans (first name hr hn hc)

private theorem stable_set (env : HeapCalls.Env) (name : String) (value : Mutable.Value)
    (written : name = "r" ∨ name = "num" ∨ name = "cnt") : Stable env (setLocal env name value) := by
  intro key hr hn hc
  rcases written with rfl | rfl | rfl
  · simp only [setLocal, hr, if_false]
  · simp only [setLocal, hn, if_false]
  · simp only [setLocal, hc, if_false]

private theorem count_branch (inSearch : Bool) (p : Nat) (indices : List Nat) (original address digit count : Nat)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (inputs : Inputs inSearch p indices original address env) (getCount : env "cnt" = some (number count))
    (getNumber : env "num" = some (number (candidate p indices digit)))
    (found : heap[address]? = some (.bytearray bytes))
    (inside : candidate p indices digit < bytes.length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    ∃ finalEnv, BlockContinues program env [(loopBody inSearch)[2]'(by cases inSearch <;> decide)] heap finalEnv heap ∧
      Inputs inSearch p indices original address finalEnv ∧
      finalEnv "cnt" = some (number (count + if ProjectEulerStatements.P51.isPrime (candidate p indices digit) then 1 else 0 : Nat)) ∧ Stable env finalEnv := by
  let value := candidate p indices digit
  have read : Evaluates program env (.subscript (.name "is_prime") (.name "num")) heap (number bytes[value].toNat) heap := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup inputs.getBuffer) (Evaluates.lookup getNumber)
    exact Mutable.getItem_buffer heap address value bytes found inside
  have live := correct value bytes[value] (List.getElem?_eq_getElem inside)
  by_cases prime : Nat.Prime value
  · let updated := setLocal env "cnt" (number (count+1 : Nat))
    refine ⟨updated, ?_, inputs_set inSearch _ _ _ _ _ inputs "cnt" _ (by cases inSearch <;> decide), ?_, stable_set env "cnt" _ (by simp)⟩
    · apply BlockContinues.branch read (flag := true)
        (by simp [Mutable.truth, scalarTruth, (live.mpr prime)])
      apply BlockContinues.augAssign getCount Evaluates.constant
      simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
    · simp [updated, setLocal, ProjectEulerStatements.P51.isPrime, value, prime]
  · refine ⟨env, ?_, inputs, ?_, stable_refl env⟩
    · have zero : bytes[value].toNat = 0 := by tauto
      exact BlockContinues.branch read (flag := false) (by simp [Mutable.truth, scalarTruth, zero]) .nil
    · simpa [ProjectEulerStatements.P51.isPrime, value, prime] using getCount


private theorem count_range (inSearch : Bool) (p : Nat) (indices : List Nat) (original address first remaining count : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original)
    (endpoint : first + remaining = 10) (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (inputs : Inputs inSearch p indices original address env) (getCount : env "cnt" = some (number count))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "r" first 10 1 (loopBody inSearch) []] heap finalEnv heap ∧
      Inputs inSearch p indices original address finalEnv ∧
      finalEnv "cnt" = some (number (count + ((List.range' first remaining).filter (accepted p indices)).length : Nat)) ∧ Stable env finalEnv := by
  induction remaining generalizing first count env with
  | zero =>
      refine ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil, inputs, ?_, stable_refl env⟩
      simpa using getCount
  | succ remaining ih =>
      let bound := setLocal env "r" (number first)
      have boundInputs := inputs_set inSearch p indices original address env inputs "r" (number first) (by cases inSearch <;> decide)
      have boundCount : bound "cnt" = some (number count) := by simpa [bound, setLocal] using getCount
      have test := leading_test inSearch p indices original address first bound heap boundInputs rfl
      let next := Stmt.rangeStepLoop "r" (first+1) 10 1 (loopBody inSearch) []
      by_cases skip : 0 ∈ indices ∧ first = 0
      · obtain ⟨finalEnv, executed, finalInputs, finalCount, saved⟩ := ih (first+1) count (by omega) bound boundInputs boundCount
        refine ⟨finalEnv, ?_, finalInputs, ?_, stable_trans (stable_set env "r" _ (by simp)) saved⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          apply BlockContinues.branch_next (middle := heap) (value := flag true) (flag := true)
            (by simpa [bound, skip] using test) (by simp [Mutable.truth, scalarTruth])
          exact BlockContinues.continue_loop (next := next) (rest := []) (by rfl) executed
        · have rejected : accepted p indices first = false := by simp [accepted, skip.1, skip.2]
          simpa [List.range'_succ, List.filter_cons, rejected] using finalCount
      · let assigned := setLocal bound "num" (number (candidate p indices first))
        have replacement := replacement_assignment inSearch p indices original first distinct bounded same heap bound
          boundInputs.getP rfl boundInputs.getOriginal boundInputs.getShift
        have assignedInputs := inputs_set inSearch p indices original address bound boundInputs "num" (number (candidate p indices first)) (by cases inSearch <;> decide)
        have assignedCount : assigned "cnt" = some (number count) := by simpa [assigned, setLocal] using boundCount
        have inside : candidate p indices first < bytes.length := by
          rw [size]
          exact ProjectEulerSolutions.PrimeReplacement.replacement_lt_pow p _ first (by omega)
        obtain ⟨middleEnv, iterated, middleInputs, middleCount, middleStable⟩ := count_branch inSearch p indices original address first count assigned heap bytes
          assignedInputs assignedCount rfl found inside correct
        obtain ⟨finalEnv, executed, finalInputs, finalCount, saved⟩ := ih (first+1)
          (count + if ProjectEulerStatements.P51.isPrime (candidate p indices first) then 1 else 0) (by omega) middleEnv
          middleInputs middleCount
        refine ⟨finalEnv, ?_, finalInputs, ?_,
          stable_trans (stable_set env "r" _ (by simp))
            (stable_trans (stable_set bound "num" _ (by simp)) (stable_trans middleStable saved))⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          apply BlockContinues.branch_next (middle := heap) (value := flag false) (flag := false)
            (by simpa [bound, skip] using test) (by simp [Mutable.truth, scalarTruth])
          have resume := BlockContinues.loop_resume executed
          exact replacement.append (iterated.append resume)
        · have allowed : first ≠ 0 ∨ 0 ∉ indices := by tauto
          by_cases prime : ProjectEulerStatements.P51.isPrime (candidate p indices first) = true
          · simpa [List.range'_succ, List.filter_cons, accepted, allowed, prime, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using finalCount
          · simpa [List.range'_succ, List.filter_cons, accepted, allowed, prime] using finalCount

/-- The shared ten-digit counting loop preserves the sieve and counts the reference family. -/
theorem counting_loop (inSearch : Bool) (p : Nat) (positive : 0 < p) (indices : List Nat) (original address count : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getOriginal : env (originalName inSearch) = some (number original))
    (getShift : env "shift" = some (number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat)))
    (getLead : env "lead" = some (flag (decide (0 ∈ indices))))
    (getBuffer : env "is_prime" = some (.ref address)) (getCount : env "cnt" = some (number count))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    ∃ finalEnv, BlockContinues program env [.forIn (.name "r") (.call (.name "range") [.constant (.int 10)]) (loopBody inSearch) []] heap finalEnv heap ∧
      finalEnv "cnt" = some (number (count + ProjectEulerStatements.P51.primeFamilySize p
        (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices) : Nat)) ∧ Stable env finalEnv := by
  obtain ⟨finalEnv, executed, _, finalCount, stable⟩ := count_range inSearch p indices original address 0 10 count distinct bounded same rfl
    env heap bytes ⟨getP, getOriginal, getShift, getLead, getBuffer⟩ getCount found size correct
  refine ⟨finalEnv, BlockContinues.for_range (.cons Evaluates.constant .nil) rfl executed, ?_, stable⟩
  have family := ProjectEulerSolutions.PrimeReplacement.familySize_reflected p positive indices original distinct bounded same
  have values (digit : Nat) :
      (((p : Int) + ((digit : Int) - original) *
        ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat)).toNat) = candidate p indices digit := by
    have equal := ProjectEulerSolutions.PrimeReplacement.replacement_value_reflected (Nat.digits 10 p) indices original digit distinct bounded same
    rw [Nat.ofDigits_digits] at equal
    rw [← equal]
    rfl
  simp only [values, List.filter_filter] at family
  have counted : ProjectEulerStatements.P51.primeFamilySize p
      (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices) =
      ((List.range 10).filter (accepted p indices)).length := by
    simpa only [accepted, Bool.and_comm] using family
  simpa only [← List.range_eq_range', ← counted] using finalCount

end replacementCounter

namespace count_prime_family_for_positions
private def code : List Stmt := executableBody ((findFunctionIn program.body "count_prime_family_for_positions").getD ([], [])).2
private def loopBody : List Stmt := match (code[11]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []

/-- The actual source assignment computes the reference replacement at reflected digit positions. -/
theorem replacement_assignment (p : Nat) (indices : List Nat) (original digit : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (getP : env "p" = some (number p))
    (getDigit : env "r" = some (number digit)) (getOriginal : env "d0" = some (number original))
    (getShift : env "shift" = some (number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat))) :
    BlockContinues program env [loopBody[1]'(by decide)] heap
      (setLocal env "num" (number (Nat.ofDigits 10 (ProjectEulerStatements.P51.replaceAt (Nat.digits 10 p)
        (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices) digit) : Nat))) heap := by
  exact replacementCounter.replacement_assignment false p indices original digit distinct bounded same heap env getP getDigit getOriginal getShift

/-- The source's complete ten-digit counting loop preserves the sieve and counts the reference family. -/
theorem counting_loop (p : Nat) (positive : 0 < p) (indices : List Nat) (original address count : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getOriginal : env "d0" = some (number original))
    (getShift : env "shift" = some (number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat)))
    (getLead : env "lead" = some (flag (decide (0 ∈ indices))))
    (getBuffer : env "is_prime" = some (.ref address)) (getCount : env "cnt" = some (number count))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    ∃ finalEnv, BlockContinues program env [code[11]'(by decide)] heap finalEnv heap ∧
      finalEnv "cnt" = some (number (count + ProjectEulerStatements.P51.primeFamilySize p
        (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices) : Nat)) := by
  obtain ⟨finalEnv, executed, counted, _⟩ := replacementCounter.counting_loop false p positive indices original address count
    distinct bounded same env heap bytes getP getOriginal getShift getLead getBuffer getCount found size correct
  exact ⟨finalEnv, executed, counted⟩

end count_prime_family_for_positions

namespace solve
private def code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private def digitBody : List Stmt := match (code[0]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def primeBody : List Stmt := match (digitBody[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def positionBody : List Stmt := match (primeBody[5]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def maskBody : List Stmt := match (positionBody[3]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def slotBody : List Stmt := match (maskBody[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def slotCondition : Expr := match (slotBody[0]? : Option Stmt) with
  | some (.ifThen condition _ _) => condition | _ => .constant .none

/-- The source subset-size assignment counts exactly the selected mask positions. -/
theorem mask_cardinality_assignment (bits width : Nat) (bounded : bits < 2 ^ width)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getMask : env "mask" = some (number bits)) :
    BlockContinues program env [maskBody[0]'(by decide)] heap
      (setLocal env "k" (number (ProjectEulerSolutions.BitSubsets.positions bits width).length)) heap := by
  apply BlockContinues.assign
  apply Evaluates.builtin (.cons (Evaluates.lookup getMask) .nil)
  change scalarResult (some (Math.bitCountCall (.int bits))) heap = _
  simp only [Math.bitCountCall, scalarNumber]
  rw [ProjectEulerSolutions.BitSubsets.bitCount_positions bits width bounded]
  rfl

/-- Testing a mask slot in the actual Python expression reads the corresponding natural-number bit. -/
theorem mask_slot_evaluates (bits index : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getMask : env "mask" = some (number bits)) (getIndex : env "j" = some (number index)) :
    Evaluates program env slotCondition
      heap (number ((bits / 2 ^ index) % 2 : Nat)) heap := by
  apply Evaluates.binary (av := number (bits / 2 ^ index : Nat)) (bv := number 1)
      (middle := heap) (afterRight := heap)
  · apply Evaluates.binary (av := number bits) (bv := number index)
      (middle := heap) (afterRight := heap) (Evaluates.lookup getMask) (Evaluates.lookup getIndex)
    simp only [Mutable.binary, scalarShiftRight, scalarNumber,
      show ¬ (index : Int) < 0 by omega, if_false, Int.toNat_natCast, integerShiftRight_nat]
    rfl
  · exact Evaluates.constant
  · simp only [Mutable.binary, scalarBitAnd, scalarNumber, integerBitAnd_nat_one]
    rfl

/-- The replacement loop inside `solve()` counts the reference family and preserves all other locals. -/
theorem family_counting_loop (p : Nat) (positive : 0 < p) (indices : List Nat) (original address count : Nat)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getOriginal : env "d" = some (number original))
    (getShift : env "shift" = some (number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat)))
    (getLead : env "lead" = some (flag (decide (0 ∈ indices))))
    (getBuffer : env "is_prime" = some (.ref address)) (getCount : env "cnt" = some (number count))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    ∃ finalEnv, BlockContinues program env [maskBody[6]'(by decide)] heap finalEnv heap ∧
      finalEnv "cnt" = some (number (count + ProjectEulerStatements.P51.primeFamilySize p
        (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices) : Nat)) ∧ (∀ name, name ≠ "r" → name ≠ "num" → name ≠ "cnt" → finalEnv name = env name) := by
  exact replacementCounter.counting_loop true p positive indices original address count distinct bounded same env heap bytes
    getP getOriginal getShift getLead getBuffer getCount found size correct

private abbrev familySize (p : Nat) (indices : List Nat) : Nat :=
  ProjectEulerStatements.P51.primeFamilySize p (ProjectEulerSolutions.PrimeReplacement.reflected (Nat.digits 10 p).length indices)

/-- The actual counter reset, replacement loop, and threshold test return exactly on a qualifying family. -/
theorem family_counting_tail (p : Nat) (positive : 0 < p) (indices : List Nat) (original address : Nat) (target : Int)
    (distinct : indices.Nodup) (bounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getOriginal : env "d" = some (number original))
    (getShift : env "shift" = some (number ((indices.map (fun index => 10 ^ ((Nat.digits 10 p).length - 1 - index))).sum : Nat)))
    (getLead : env "lead" = some (flag (decide (0 ∈ indices))))
    (getBuffer : env "is_prime" = some (.ref address)) (getTarget : env "target" = some (number target))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    (target ≤ (familySize p indices : Int) → BlockReturns program env (maskBody.drop 5) heap (number p) heap) ∧
    ((familySize p indices : Int) < target → ∃ finalEnv,
      BlockContinues program env (maskBody.drop 5) heap finalEnv heap ∧
      finalEnv "cnt" = some (number (familySize p indices)) ∧
      (∀ name, name ≠ "r" → name ≠ "num" → name ≠ "cnt" → finalEnv name = env name)) := by
  let reset := setLocal env "cnt" (number 0)
  have initialized : BlockContinues program env [maskBody[5]'(by decide)] heap reset heap :=
    BlockContinues.assign Evaluates.constant
  obtain ⟨finalEnv, counted, total, preserved⟩ := family_counting_loop p positive indices original address 0 distinct bounded same
    reset heap bytes (by simpa [reset, setLocal] using getP) (by simpa [reset, setLocal] using getOriginal)
    (by simpa [reset, setLocal] using getShift) (by simpa [reset, setLocal] using getLead)
    (by simpa [reset, setLocal] using getBuffer) rfl found size correct
  have getTotal : finalEnv "cnt" = some (number (familySize p indices)) := by simpa only [Nat.zero_add] using total
  have saved : ∀ name, name ≠ "r" → name ≠ "num" → name ≠ "cnt" → finalEnv name = env name := by
    intro name hr hn hc
    exact (preserved name hr hn hc).trans (by simp only [reset, setLocal, hc, if_false])
  have finalP : finalEnv "p" = some (number p) := (saved "p" (by decide) (by decide) (by decide)).trans getP
  have finalTarget : finalEnv "target" = some (number target) := (saved "target" (by decide) (by decide) (by decide)).trans getTarget
  have test : Evaluates program finalEnv (.compare (.name "cnt") [(.gtE, .name "target")]) heap
      (flag (decide (target ≤ (familySize p indices : Int)))) heap := by
    apply Evaluates.compare (av := number (familySize p indices)) (bv := number target)
      (middle := heap) (afterRight := heap) (Evaluates.lookup getTotal) (Evaluates.lookup finalTarget)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  constructor
  · intro qualifies
    have returned : BlockReturns program finalEnv [maskBody[7]'(by decide)] heap (number p) heap := by
      apply BlockReturns.branch (value := flag true) (flag := true)
        (by simpa only [qualifies, decide_true] using test) (by simp [Mutable.truth, scalarTruth])
      exact BlockReturns.return_value (Evaluates.lookup finalP)
    exact (initialized.append counted).returns returned
  · intro tooSmall
    refine ⟨finalEnv, ?_, getTotal, saved⟩
    have rejected : BlockContinues program finalEnv [maskBody[7]'(by decide)] heap finalEnv heap := by
      apply BlockContinues.branch (value := flag false) (flag := false)
        (by simpa only [show ¬ target ≤ (familySize p indices : Int) by omega, decide_false] using test)
        (by simp [Mutable.truth, scalarTruth])
      exact .nil
    exact initialized.append (counted.append rejected)

end solve

end solvers.«51»
