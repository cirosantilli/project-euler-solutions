import «.formalsw».solvers.«51.py.ast»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.LoopCalls
import ProjectEulerSolutions.Lib.PrimeSieve
import ProjectEulerSolutions.Lib.PrimeReplacement
import ProjectEulerSolutions.Lib.PrimeReplacementSearch
import ProjectEulerSolutions.Lib.BitSubsets
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.HeapRows
import Mathlib.Data.Nat.Sqrt
import ProjectEulerStatements.Lib.DecimalDigits
import data.«project-euler-statements».data.py.project_euler_statements.«p51.py»

/-! Source-level termination and equivalence proofs for problem 51's complete prime-family search. -/

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
    (target ≤ (familySize p indices : Int) → ∀ following, BlockReturns program env (maskBody.drop 5 ++ following) heap (number p) heap) ∧
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
  · intro qualifies following
    have returned : BlockReturns program finalEnv ([maskBody[7]'(by decide)] ++ following) heap (number p) heap := by
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

private def numbers (items : List Nat) : List Mutable.Value := items.map (fun (item : Nat) => number item)
private def powers (width : Nat) : List Nat := (List.range width).map (fun index => 10 ^ (width - 1 - index))

private theorem read_natural_list (env : HeapCalls.Env) (heap : Mutable.Heap) (arrayName indexName : String)
    (address index : Nat) (items : List Nat) (inside : index < items.length)
    (getArray : env arrayName = some (.ref address)) (getIndex : env indexName = some (number index))
    (found : heap[address]? = some (.list (numbers items))) :
    Evaluates program env (.subscript (.name arrayName) (.name indexName)) heap (number items[index]) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getArray) (Evaluates.lookup getIndex)
  simpa only [numbers, List.getElem_map] using Mutable.getItem_list heap address index (numbers items) found
    (by simpa only [numbers, List.length_map] using inside)

private theorem mask_slot_truth (bits index : Nat) (heap : Mutable.Heap) :
    Mutable.truth (number ((bits / 2 ^ index) % 2 : Nat)) heap = some (.ok (bits.testBit index), heap) := by
  have parity : (bits / 2 ^ index) % 2 = 0 ∨ (bits / 2 ^ index) % 2 = 1 := by omega
  rcases parity with zero | one
  · simp [Mutable.truth, scalarTruth, Nat.testBit_eq_decide_div_mod_eq, zero]
  · simp [Mutable.truth, scalarTruth, Nat.testBit_eq_decide_div_mod_eq, one]


namespace slotWeights
private def yesBody : List Stmt := match (slotBody[0]? : Option Stmt) with
  | some (.ifThen _ yes _) => yes | _ => []
private def weight (width index : Nat) : Nat := 10 ^ (width - 1 - index)
private def selected (items : List Nat) (bits first remaining : Nat) : List Nat :=
  ((List.range' first remaining).filter bits.testBit).map (fun index => items[index]!)
private def total (width : Nat) (indices : List Nat) : Nat := (indices.map (weight width)).sum

private structure Inputs (bits posAddress powerAddress : Nat) (env : HeapCalls.Env) : Prop where
  getMask : env "mask" = some (number bits)
  getPositions : env "pos" = some (.ref posAddress)
  getPowers : env "pow10" = some (.ref powerAddress)

private theorem inputs_set (bits posAddress powerAddress : Nat) (env : HeapCalls.Env)
    (inputs : Inputs bits posAddress powerAddress env) (name : String) (value : Mutable.Value)
    (different : name ≠ "mask" ∧ name ≠ "pos" ∧ name ≠ "pow10") :
    Inputs bits posAddress powerAddress (setLocal env name value) := by
  exact ⟨by simpa [setLocal, Ne.symm different.1] using inputs.getMask,
    by simpa [setLocal, Ne.symm different.2.1] using inputs.getPositions,
    by simpa [setLocal, Ne.symm different.2.2] using inputs.getPowers⟩

private def Stable (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "j" → name ≠ "idx" → name ≠ "shift" → name ≠ "lead" → after name = before name

private theorem stable_refl (env : HeapCalls.Env) : Stable env env := fun _ _ _ _ _ => rfl
private theorem stable_trans (first : Stable before middle) (second : Stable middle after) : Stable before after :=
  fun name hj hi hs hl => (second name hj hi hs hl).trans (first name hj hi hs hl)
private theorem stable_set (env : HeapCalls.Env) (name : String) (value : Mutable.Value)
    (written : name = "j" ∨ name = "idx" ∨ name = "shift" ∨ name = "lead") : Stable env (setLocal env name value) := by
  intro key hj hi hs hl
  rcases written with rfl | rfl | rfl | rfl
  · simp only [setLocal, hj, if_false]
  · simp only [setLocal, hi, if_false]
  · simp only [setLocal, hs, if_false]
  · simp only [setLocal, hl, if_false]

private theorem iteration (width bits posAddress powerAddress first accumulated : Nat) (leading : Bool)
    (items : List Nat) (inside : first < items.length) (bounded : ∀ index ∈ items, index < width)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (inputs : Inputs bits posAddress powerAddress env)
    (getIndex : env "j" = some (number first)) (getShift : env "shift" = some (number accumulated))
    (getLead : env "lead" = some (flag leading))
    (positionsFound : heap[posAddress]? = some (.list (numbers items)))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers width)))) :
    ∃ finalEnv, BlockContinues program env yesBody heap finalEnv heap ∧
      Inputs bits posAddress powerAddress finalEnv ∧
      finalEnv "shift" = some (number (accumulated + weight width items[first] : Nat)) ∧
      finalEnv "lead" = some (flag (leading || decide (items[first] = 0))) ∧ Stable env finalEnv := by
  let index := items[first]
  have indexBound : index < width := bounded index (List.getElem_mem inside)
  let located := setLocal env "idx" (number index)
  let weighted := setLocal located "shift" (number (accumulated + weight width index : Nat))
  have locate : BlockContinues program env [yesBody[0]'(by decide)] heap located heap :=
    BlockContinues.assign (read_natural_list env heap "pos" "j" posAddress first items inside inputs.getPositions getIndex positionsFound)
  have getLocated : located "idx" = some (number index) := by simp [located, setLocal]
  have readPower := read_natural_list located heap "pow10" "idx" powerAddress index (powers width)
    (by simpa only [powers, List.length_map, List.length_range] using indexBound)
    (by simpa [located, setLocal] using inputs.getPowers) getLocated powersFound
  have add : BlockContinues program located [yesBody[1]'(by decide)] heap weighted heap := by
    apply BlockContinues.augAssign (by simpa [located, setLocal] using getShift)
      (by simpa only [powers, List.getElem_map, List.getElem_range] using readPower)
    simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number, weight]
  have weightedInputs : Inputs bits posAddress powerAddress weighted :=
    inputs_set _ _ _ _ (inputs_set _ _ _ _ inputs "idx" _ (by decide)) "shift" _ (by decide)
  have prefixStable : Stable env weighted := stable_trans (stable_set env "idx" _ (by simp)) (stable_set located "shift" _ (by simp))
  have test : Evaluates program weighted (.compare (.name "idx") [(.eq, .constant (.int 0))]) heap
      (flag (decide (index = 0))) heap := by
    apply Evaluates.compare (av := number index) (bv := number 0) (middle := heap) (afterRight := heap)
      (Evaluates.lookup (by simp [weighted, located, setLocal])) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases zero : index = 0
  · have actualZero : items[first] = 0 := zero
    let marked := setLocal weighted "lead" (flag true)
    have mark : BlockContinues program weighted [yesBody[2]'(by decide)] heap marked heap := by
      apply BlockContinues.branch (value := flag true) (flag := true)
        (by simpa only [zero, decide_true] using test) (by simp [Mutable.truth, scalarTruth])
      exact BlockContinues.assign Evaluates.constant
    refine ⟨marked, locate.append (add.append mark), inputs_set _ _ _ _ weightedInputs "lead" _ (by decide), ?_, ?_,
      stable_trans prefixStable (stable_set weighted "lead" _ (by simp))⟩
    · simp [marked, weighted, setLocal, index]
    · simp [marked, setLocal, actualZero]
  · have actualNonzero : items[first] ≠ 0 := zero
    have keep : BlockContinues program weighted [yesBody[2]'(by decide)] heap weighted heap :=
      BlockContinues.branch (flag := false) (by simpa only [zero, decide_false] using test)
        (by simp [Mutable.truth, scalarTruth]) .nil
    refine ⟨weighted, locate.append (add.append keep), weightedInputs, by simp [weighted, setLocal, index], ?_, prefixStable⟩
    simpa [weighted, located, setLocal, actualNonzero] using getLead

private theorem loop (width bits posAddress powerAddress first remaining accumulated : Nat) (leading : Bool)
    (items : List Nat) (endpoint : first + remaining = items.length)
    (bounded : ∀ index ∈ items, index < width) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (inputs : Inputs bits posAddress powerAddress env)
    (getShift : env "shift" = some (number accumulated)) (getLead : env "lead" = some (flag leading))
    (positionsFound : heap[posAddress]? = some (.list (numbers items)))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers width)))) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "j" first items.length 1 slotBody []] heap finalEnv heap ∧
      Inputs bits posAddress powerAddress finalEnv ∧
      finalEnv "shift" = some (number (accumulated + total width (selected items bits first remaining) : Nat)) ∧
      finalEnv "lead" = some (flag (leading || decide (0 ∈ selected items bits first remaining))) ∧ Stable env finalEnv := by
  induction remaining generalizing first accumulated leading env with
  | zero =>
      refine ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil, inputs, ?_, ?_, stable_refl env⟩
      · simpa [selected, total] using getShift
      · simpa [selected] using getLead
  | succ remaining ih =>
      have inside : first < items.length := by omega
      let bound := setLocal env "j" (number first)
      have boundInputs := inputs_set _ _ _ _ inputs "j" (number first) (by decide)
      have boundShift : bound "shift" = some (number accumulated) := by simpa [bound, setLocal] using getShift
      have boundLead : bound "lead" = some (flag leading) := by simpa [bound, setLocal] using getLead
      have test := mask_slot_evaluates bits first bound heap boundInputs.getMask rfl
      have truth := mask_slot_truth bits first heap
      cases hit : bits.testBit first with
      | false =>
          obtain ⟨finalEnv, executed, finalInputs, finalShift, finalLead, saved⟩ :=
            ih (first+1) accumulated leading (by omega) bound boundInputs boundShift boundLead
          have skipped : BlockContinues program bound slotBody heap bound heap :=
            BlockContinues.branch (flag := false) test (by simpa only [hit] using truth) .nil
          refine ⟨finalEnv, ?_, finalInputs, ?_, ?_, stable_trans (stable_set env "j" _ (by simp)) saved⟩
          · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
            exact skipped.append executed
          · simpa only [selected, List.range'_succ, List.filter_cons, hit, Bool.false_eq_true, if_false] using finalShift
          · simpa only [selected, List.range'_succ, List.filter_cons, hit, Bool.false_eq_true, if_false] using finalLead
      | true =>
          obtain ⟨middleEnv, iterated, middleInputs, middleShift, middleLead, middleStable⟩ :=
            iteration width bits posAddress powerAddress first accumulated leading items inside bounded bound heap
              boundInputs rfl boundShift boundLead positionsFound powersFound
          obtain ⟨finalEnv, executed, finalInputs, finalShift, finalLead, saved⟩ :=
            ih (first+1) (accumulated + weight width items[first]) (leading || decide (items[first] = 0)) (by omega)
              middleEnv middleInputs middleShift middleLead
          have entered : BlockContinues program bound slotBody heap middleEnv heap :=
            BlockContinues.branch (flag := true) test (by simpa only [hit] using truth) iterated
          have read : items[first]! = items[first] := List.getElem!_of_getElem? (List.getElem?_eq_getElem inside)
          refine ⟨finalEnv, ?_, finalInputs, ?_, ?_,
            stable_trans (stable_set env "j" _ (by simp)) (stable_trans middleStable saved)⟩
          · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
            exact entered.append executed
          · simpa only [selected, List.range'_succ, List.filter_cons, hit, Bool.true_eq, if_true, List.map_cons, total, List.sum_cons, read, Nat.add_assoc] using finalShift
          · have zeroEq : (0 = items[first]) ↔ (items[first] = 0) := eq_comm
            simpa only [selected, List.range'_succ, List.filter_cons, hit, Bool.true_eq, if_true, List.map_cons, read,
              List.mem_cons, zeroEq, Bool.decide_or, Bool.or_assoc] using finalLead

/-- The source's position loop sums exactly the selected decimal weights and detects leading replacement. -/
theorem result (width bits posAddress powerAddress : Nat) (items : List Nat)
    (bounded : ∀ index ∈ items, index < width) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getMask : env "mask" = some (number bits)) (getPositions : env "pos" = some (.ref posAddress))
    (getPowers : env "pow10" = some (.ref powerAddress)) (getLength : env "m" = some (number items.length))
    (positionsFound : heap[posAddress]? = some (.list (numbers items)))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers width)))) :
    ∃ finalEnv, BlockContinues program env ((maskBody.drop 2).take 3) heap finalEnv heap ∧
      finalEnv "shift" = some (number (((ProjectEulerSolutions.BitSubsets.select items bits).map (fun index => 10 ^ (width - 1 - index))).sum : Nat)) ∧
      finalEnv "lead" = some (flag (decide (0 ∈ ProjectEulerSolutions.BitSubsets.select items bits))) ∧
      (∀ name, name ≠ "j" → name ≠ "idx" → name ≠ "shift" → name ≠ "lead" → finalEnv name = env name) := by
  let cleared := setLocal env "shift" (number 0)
  let initialized := setLocal cleared "lead" (flag false)
  have clear : BlockContinues program env [maskBody[2]'(by decide)] heap cleared heap := BlockContinues.assign Evaluates.constant
  have initiallyFalse : BlockContinues program cleared [maskBody[3]'(by decide)] heap initialized heap := BlockContinues.assign Evaluates.constant
  have inputs : Inputs bits posAddress powerAddress initialized := by
    exact ⟨by simpa [initialized, cleared, setLocal] using getMask,
      by simpa [initialized, cleared, setLocal] using getPositions,
      by simpa [initialized, cleared, setLocal] using getPowers⟩
  obtain ⟨finalEnv, executed, _, finalShift, finalLead, saved⟩ := loop width bits posAddress powerAddress 0 items.length 0 false items (by omega) bounded
    initialized heap inputs (by simp [initialized, cleared, setLocal]) rfl positionsFound powersFound
  have ranged : BlockContinues program initialized [maskBody[4]'(by decide)] heap finalEnv heap := by
    apply BlockContinues.for_range (.cons (Evaluates.lookup (by simpa [initialized, cleared, setLocal] using getLength)) .nil) rfl executed
  refine ⟨finalEnv, clear.append (initiallyFalse.append ranged), ?_, ?_, ?_⟩
  · simpa only [Nat.zero_add, total, selected, ← List.range_eq_range', ProjectEulerSolutions.BitSubsets.select,
      ProjectEulerSolutions.BitSubsets.positions, weight] using finalShift
  · simpa only [Bool.false_or, selected, ← List.range_eq_range', ProjectEulerSolutions.BitSubsets.select,
      ProjectEulerSolutions.BitSubsets.positions] using finalLead
  · exact stable_trans (stable_set env "shift" _ (by simp)) (stable_trans (stable_set cleared "lead" _ (by simp)) saved)

end slotWeights

/-- After pruning, one complete source mask iteration returns exactly for a qualifying replacement family. -/
theorem mask_tail (p : Nat) (positive : 0 < p) (items : List Nat) (original bits posAddress powerAddress primeAddress : Nat)
    (target : Int) (distinct : items.Nodup) (bounded : ∀ index ∈ items, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse items original)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getOriginal : env "d" = some (number original))
    (getMask : env "mask" = some (number bits)) (getPositions : env "pos" = some (.ref posAddress))
    (getPowers : env "pow10" = some (.ref powerAddress)) (getLength : env "m" = some (number items.length))
    (getPrimes : env "is_prime" = some (.ref primeAddress)) (getTarget : env "target" = some (number target))
    (positionsFound : heap[posAddress]? = some (.list (numbers items)))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers (Nat.digits 10 p).length))))
    (primesFound : heap[primeAddress]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    (target ≤ (familySize p (ProjectEulerSolutions.BitSubsets.select items bits) : Int) →
      ∀ following, BlockReturns program env (maskBody.drop 2 ++ following) heap (number p) heap) ∧
    ((familySize p (ProjectEulerSolutions.BitSubsets.select items bits) : Int) < target → ∃ finalEnv,
      BlockContinues program env (maskBody.drop 2) heap finalEnv heap ∧
      finalEnv "cnt" = some (number (familySize p (ProjectEulerSolutions.BitSubsets.select items bits))) ∧
      (∀ name, name ∉ ["j", "idx", "shift", "lead", "r", "num", "cnt"] → finalEnv name = env name)) := by
  let indices := ProjectEulerSolutions.BitSubsets.select items bits
  have included := ProjectEulerSolutions.BitSubsets.select_subset items bits
  have selectedDistinct := ProjectEulerSolutions.BitSubsets.select_nodup items distinct bits
  have selectedBounded : ∀ index ∈ indices, index < (Nat.digits 10 p).length :=
    fun index member => bounded index (included member)
  have selectedSame : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse indices original := by
    intro index value found member
    exact same index value found (by simpa only [Nat.zero_add] using included (by simpa only [Nat.zero_add] using member))
  obtain ⟨preparedEnv, prepared, getShift, getLead, saved⟩ := slotWeights.result (Nat.digits 10 p).length bits posAddress powerAddress
    items bounded env heap getMask getPositions getPowers getLength positionsFound powersFound
  have preparedP := (saved "p" (by decide) (by decide) (by decide) (by decide)).trans getP
  have preparedOriginal := (saved "d" (by decide) (by decide) (by decide) (by decide)).trans getOriginal
  have preparedPrimes := (saved "is_prime" (by decide) (by decide) (by decide) (by decide)).trans getPrimes
  have preparedTarget := (saved "target" (by decide) (by decide) (by decide) (by decide)).trans getTarget
  have counted := family_counting_tail p positive indices original primeAddress target selectedDistinct selectedBounded selectedSame
    preparedEnv heap bytes preparedP preparedOriginal getShift getLead preparedPrimes preparedTarget primesFound size correct
  constructor
  · intro qualifies following
    exact prepared.returns (counted.1 qualifies following)
  · intro tooSmall
    obtain ⟨finalEnv, executed, total, preserved⟩ := counted.2 tooSmall
    refine ⟨finalEnv, prepared.append executed, total, ?_⟩
    intro name untouched
    have hj : name ≠ "j" := fun equal => untouched (by simp [equal])
    have hi : name ≠ "idx" := fun equal => untouched (by simp [equal])
    have hs : name ≠ "shift" := fun equal => untouched (by simp [equal])
    have hl : name ≠ "lead" := fun equal => untouched (by simp [equal])
    have hr : name ≠ "r" := fun equal => untouched (by simp [equal])
    have hn : name ≠ "num" := fun equal => untouched (by simp [equal])
    have hc : name ≠ "cnt" := fun equal => untouched (by simp [equal])
    exact (preserved name hr hn hc).trans (saved name hj hi hs hl)

namespace masks
private def guard : Expr := match (maskBody[1]? : Option Stmt) with
  | some (.ifThen condition _ _) => condition | _ => .constant .none

private theorem guard_evaluates (target : Int) (count : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) (getCount : env "k" = some (number count)) :
    Evaluates program env guard heap (flag (decide (8 ≤ target ∧ count % 3 ≠ 0))) heap := by
  have threshold : Evaluates program env (.compare (.name "target") [(.gtE, .constant (.int 8))]) heap
      (flag (decide (8 ≤ target))) heap := by
    apply Evaluates.compare (av := number target) (bv := number 8) (middle := heap) (afterRight := heap)
      (Evaluates.lookup getTarget) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases large : 8 ≤ target
  · apply Evaluates.boolean_next (value := flag true) (flag := true)
      (by simpa only [large, decide_true] using threshold) (by simp [Mutable.truth, scalarTruth]) (by decide)
    apply Evaluates.boolean_last (flag := decide (count % 3 ≠ 0))
    · apply Evaluates.compare (av := number (count % 3 : Nat)) (bv := number 0) (middle := heap) (afterRight := heap)
      · exact Evaluates.binary (av := number count) (bv := number 3) (middle := heap) (afterRight := heap)
          (Evaluates.lookup getCount) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
      · exact Evaluates.constant
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp, large]
        omega
    · simp [Mutable.truth, scalarTruth, large]
  · simpa [guard, large] using Evaluates.boolean_stop (op := BoolOp.and)
      (rest := [.compare (.binOp (.name "k") .mod (.constant (.int 3))) [(.notEq, .constant (.int 0))]])
      (value := flag false) (flag := false) (by simpa only [large, decide_false] using threshold)
      (by simp [Mutable.truth, scalarTruth]) (by decide)

private def Run (p : Nat) (invariant : HeapCalls.Env → Prop) (env : HeapCalls.Env)
    (body : List Stmt) (heap : Mutable.Heap) (found : Bool) : Prop :=
  if found then ∀ following, BlockReturns program env (body ++ following) heap (number p) heap
  else ∃ finalEnv, BlockContinues program env body heap finalEnv heap ∧ invariant finalEnv

private theorem run_weaken (implication : ∀ finalEnv, invariant finalEnv → weaker finalEnv)
    (executed : Run p invariant env body heap found) : Run p weaker env body heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, continued, valid⟩ := executed
      exact ⟨finalEnv, continued, implication finalEnv valid⟩
  | true => exact executed

private theorem run_prepend (continued : BlockContinues program env body heap middleEnv heap)
    (rest : Run p invariant middleEnv following heap found) : Run p invariant env (body ++ following) heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, finished, valid⟩ := rest
      exact ⟨finalEnv, continued.append finished, valid⟩
  | true =>
      intro after
      simpa only [List.append_assoc] using continued.returns (rest after)

private theorem run_branch {flag : Bool} (evaluated : Evaluates program env condition heap value heap)
    (truth : Mutable.truth value heap = some (.ok flag, heap))
    (executed : Run p invariant env ((if flag then yes else no) ++ rest) heap found) :
    Run p invariant env (.ifThen condition yes no :: rest) heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, continued, valid⟩ := executed
      exact ⟨finalEnv, BlockContinues.branch_next evaluated truth continued, valid⟩
  | true =>
      intro after
      exact BlockReturns.branch evaluated truth (by simpa only [List.append_assoc] using executed after)

private theorem run_resume (executed : Run p invariant env [next] heap found) :
    Run p invariant env [.loopResume next] heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, continued, valid⟩ := executed
      exact ⟨finalEnv, BlockContinues.loop_resume continued, valid⟩
  | true =>
      intro after
      exact BlockReturns.loop_resume (executed after)

private theorem run_continue (target : Calls.loopTarget body = some (next, rest))
    (executed : Run p invariant env (next :: rest) heap found) :
    Run p invariant env (.continueLoop :: body) heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, continued, valid⟩ := executed
      exact ⟨finalEnv, BlockContinues.continue_loop target continued, valid⟩
  | true =>
      intro after
      exact BlockReturns.continue_loop (Calls.loopTarget_append target after) (by simpa only [List.cons_append] using executed after)

private theorem run_range_step (active : Calls.rangeContinues first stop stride = true)
    (rest : Run p invariant (setLocal env name (number first))
      (body ++ [Calls.loopContinuation body (.rangeStepLoop name (first+stride) stop stride body [])]) heap found) :
    Run p invariant env [.rangeStepLoop name first stop stride body []] heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, continued, valid⟩ := rest
      exact ⟨finalEnv, BlockContinues.range_step active continued, valid⟩
  | true =>
      intro after
      apply BlockReturns.range_step active
      simpa only [List.append_assoc] using rest after

private def writes : List String := ["mask", "k"] ++ ["j", "idx", "shift", "lead", "r", "num", "cnt"]
private def Stable (before after : HeapCalls.Env) : Prop := ∀ name, name ∉ writes → after name = before name
private theorem stable_refl (env : HeapCalls.Env) : Stable env env := fun _ _ => rfl
private theorem stable_trans (first : Stable before middle) (second : Stable middle after) : Stable before after :=
  fun name untouched => (second name untouched).trans (first name untouched)
private theorem stable_set (env : HeapCalls.Env) (name : String) (value : Mutable.Value) (written : name ∈ writes) :
    Stable env (setLocal env name value) := by
  intro key untouched
  have different : key ≠ name := by intro same; subst key; exact untouched written
  simp only [setLocal, different, if_false]

private structure Inputs (p original posAddress powerAddress primeAddress : Nat) (target : Int) (items : List Nat)
    (env : HeapCalls.Env) : Prop where
  getP : env "p" = some (number p)
  getOriginal : env "d" = some (number original)
  getPositions : env "pos" = some (.ref posAddress)
  getPowers : env "pow10" = some (.ref powerAddress)
  getLength : env "m" = some (number items.length)
  getPrimes : env "is_prime" = some (.ref primeAddress)
  getTarget : env "target" = some (number target)

private theorem Inputs.preserved (inputs : Inputs p original posAddress powerAddress primeAddress target items before)
    (stable : Stable before after) : Inputs p original posAddress powerAddress primeAddress target items after := by
  exact ⟨(stable "p" (by decide)).trans inputs.getP, (stable "d" (by decide)).trans inputs.getOriginal,
    (stable "pos" (by decide)).trans inputs.getPositions, (stable "pow10" (by decide)).trans inputs.getPowers,
    (stable "m" (by decide)).trans inputs.getLength, (stable "is_prime" (by decide)).trans inputs.getPrimes,
    (stable "target" (by decide)).trans inputs.getTarget⟩

private def accepted (target : Int) (p : Nat) (items : List Nat) (bits : Nat) : Bool :=
  if 8 ≤ target ∧ FormalSW.Py.Builtin.Int.bitCount bits % 3 ≠ 0 then false
  else decide (target ≤ (familySize p (ProjectEulerSolutions.BitSubsets.select items bits) : Int))

private theorem loop (p original posAddress powerAddress primeAddress first remaining stop : Nat) (target : Int)
    (items : List Nat) (positive : 0 < p) (distinct : items.Nodup)
    (bounded : ∀ index ∈ items, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse items original)
    (endpoint : first + remaining = stop)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (inputs : Inputs p original posAddress powerAddress primeAddress target items env)
    (positionsFound : heap[posAddress]? = some (.list (numbers items)))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers (Nat.digits 10 p).length))))
    (primesFound : heap[primeAddress]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Run p (fun finalEnv => Inputs p original posAddress powerAddress primeAddress target items finalEnv ∧ Stable env finalEnv)
      env [.rangeStepLoop "mask" first stop 1 maskBody []] heap ((List.range' first remaining).any (accepted target p items)) := by
  induction remaining generalizing first env with
  | zero =>
      refine ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil, inputs, stable_refl env⟩
  | succ remaining ih =>
      let bound := setLocal env "mask" (number first)
      let keyed := setLocal bound "k" (number (FormalSW.Py.Builtin.Int.bitCount first))
      let next := Stmt.rangeStepLoop "mask" (first+1) stop 1 maskBody []
      have prefixStable : Stable env keyed := stable_trans (stable_set env "mask" _ (by decide)) (stable_set bound "k" _ (by decide))
      have keyedInputs := inputs.preserved prefixStable
      have assigned : BlockContinues program bound [maskBody[0]'(by decide)] heap keyed heap := by
        apply BlockContinues.assign
        apply Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
        rfl
      have test := guard_evaluates target (FormalSW.Py.Builtin.Int.bitCount first) keyed heap keyedInputs.getTarget rfl
      by_cases skip : 8 ≤ target ∧ FormalSW.Py.Builtin.Int.bitCount first % 3 ≠ 0
      · have inherited := run_weaken
          (invariant := fun finalEnv => Inputs p original posAddress powerAddress primeAddress target items finalEnv ∧ Stable keyed finalEnv)
          (weaker := fun finalEnv => Inputs p original posAddress powerAddress primeAddress target items finalEnv ∧ Stable env finalEnv)
          (fun finalEnv valid => ⟨valid.1, stable_trans prefixStable valid.2⟩)
          (ih (first+1) (by omega) keyed keyedInputs)
        have rejected : accepted target p items first = false := by unfold accepted; rw [if_pos skip]
        rw [List.range'_succ, List.any_cons, rejected, Bool.false_or]
        apply run_range_step (by simp [Calls.rangeContinues]; omega)
        change Run p _ bound ([maskBody[0]'(by decide)] ++ (maskBody.drop 1 ++ [.loopResume next])) heap _
        apply run_prepend assigned
        apply run_branch (flag := true) (value := flag true)
          (by simpa only [show decide (8 ≤ target ∧ FormalSW.Py.Builtin.Int.bitCount first % 3 ≠ 0) = true from decide_eq_true skip] using test) (by simp [Mutable.truth, scalarTruth])
        exact run_continue (next := next) (rest := []) rfl inherited
      · have counted := mask_tail p positive items original first posAddress powerAddress primeAddress target distinct bounded same
          keyed heap bytes keyedInputs.getP keyedInputs.getOriginal (by simp [keyed, bound, setLocal])
          keyedInputs.getPositions keyedInputs.getPowers keyedInputs.getLength keyedInputs.getPrimes keyedInputs.getTarget
          positionsFound powersFound primesFound size correct
        by_cases qualifies : target ≤ (familySize p (ProjectEulerSolutions.BitSubsets.select items first) : Int)
        · have acceptedHere : accepted target p items first = true := by simp only [accepted, skip, if_false, qualifies, decide_true]
          rw [List.range'_succ, List.any_cons, acceptedHere, Bool.true_or]
          apply run_range_step (by simp [Calls.rangeContinues]; omega)
          change Run p _ bound ([maskBody[0]'(by decide)] ++ (maskBody.drop 1 ++ [.loopResume next])) heap true
          apply run_prepend assigned
          apply run_branch (flag := false) (value := flag false)
            (by simpa only [skip, decide_false] using test) (by simp [Mutable.truth, scalarTruth])
          intro following
          simpa only [List.append_assoc] using counted.1 qualifies ([.loopResume next] ++ following)
        · have tooSmall : (familySize p (ProjectEulerSolutions.BitSubsets.select items first) : Int) < target := by omega
          obtain ⟨middleEnv, continued, _, preserved⟩ := counted.2 tooSmall
          have bodyStable : Stable keyed middleEnv := by
            intro name untouched
            exact preserved name (fun member => untouched (List.mem_append_right _ member))
          have combined := stable_trans prefixStable bodyStable
          have middleInputs := inputs.preserved combined
          have inherited := run_weaken
            (invariant := fun finalEnv => Inputs p original posAddress powerAddress primeAddress target items finalEnv ∧ Stable middleEnv finalEnv)
            (weaker := fun finalEnv => Inputs p original posAddress powerAddress primeAddress target items finalEnv ∧ Stable env finalEnv)
            (fun finalEnv valid => ⟨valid.1, stable_trans combined valid.2⟩)
            (ih (first+1) (by omega) middleEnv middleInputs)
          have rejected : accepted target p items first = false := by simp only [accepted, skip, if_false, qualifies, decide_false]
          rw [List.range'_succ, List.any_cons, rejected, Bool.false_or]
          apply run_range_step (by simp [Calls.rangeContinues]; omega)
          change Run p _ bound ([maskBody[0]'(by decide)] ++ (maskBody.drop 1 ++ [.loopResume next])) heap _
          apply run_prepend assigned
          apply run_branch (flag := false) (value := flag false)
            (by simpa only [skip, decide_false] using test) (by simp [Mutable.truth, scalarTruth])
          change Run p _ keyed (maskBody.drop 2 ++ [.loopResume next]) heap _
          exact run_prepend continued (run_resume inherited)

private theorem run_for_range (arguments : Evaluates.Arguments program env expressions heap values heap)
    (bounds : rangeBounds values = .ok (first, stop, stride))
    (executed : Run p invariant env [.rangeStepLoop name first stop stride body []] heap found) :
    Run p invariant env [.forIn (.name name) (.call (.name "range") expressions) body []] heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, continued, valid⟩ := executed
      exact ⟨finalEnv, BlockContinues.for_range arguments bounds continued, valid⟩
  | true =>
      intro following
      exact BlockReturns.for_range arguments bounds (executed following)

/-- The complete source mask loop searches every surviving subset and preserves the outer search state. -/
theorem result (p original posAddress powerAddress primeAddress : Nat) (target : Int)
    (items : List Nat) (positive : 0 < p) (distinct : items.Nodup)
    (bounded : ∀ index ∈ items, index < (Nat.digits 10 p).length)
    (same : ProjectEulerSolutions.PrimeReplacement.SameDigit (Nat.digits 10 p).reverse items original)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getOriginal : env "d" = some (number original))
    (getPositions : env "pos" = some (.ref posAddress)) (getPowers : env "pow10" = some (.ref powerAddress))
    (getLength : env "m" = some (number items.length)) (getPrimes : env "is_prime" = some (.ref primeAddress))
    (getTarget : env "target" = some (number target))
    (positionsFound : heap[posAddress]? = some (.list (numbers items)))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers (Nat.digits 10 p).length))))
    (primesFound : heap[primeAddress]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Run p (Stable env) env [positionBody[3]'(by decide)] heap
      ((List.range' 1 (2 ^ items.length - 1)).any (accepted target p items)) := by
  have positiveStop : 0 < 2 ^ items.length := Nat.two_pow_pos _
  have scanned := loop p original posAddress powerAddress primeAddress 1 (2 ^ items.length - 1) (2 ^ items.length) target items
    positive distinct bounded same (by omega) env heap bytes
    ⟨getP, getOriginal, getPositions, getPowers, getLength, getPrimes, getTarget⟩ positionsFound powersFound primesFound size correct
  have stop : Evaluates program env (.binOp (.constant (.int 1)) .leftShift (.name "m")) heap (number (2 ^ items.length : Nat)) heap := by
    apply Evaluates.binary (av := number 1) (bv := number items.length) (middle := heap) (afterRight := heap)
      Evaluates.constant (Evaluates.lookup getLength)
    simp only [Mutable.binary, scalarShiftLeft, scalarNumber, show ¬ (items.length : Int) < 0 by omega,
      if_false, Int.toNat_natCast, integerShiftLeft, Int.one_mul, Int.natCast_pow]
    rfl
  exact run_weaken (weaker := Stable env) (fun _ valid => valid.2)
    (run_for_range (.cons Evaluates.constant (.cons stop .nil)) rfl scanned)

private theorem accepted_nat (target : Int) (p : Nat) (items : List Nat) :
    accepted target p items = ProjectEulerSolutions.PrimeReplacementSearch.accepts target.toNat p items := by
  funext bits
  have threshold : 8 ≤ target ↔ 8 ≤ target.toNat := by omega
  have score : target ≤ (familySize p (ProjectEulerSolutions.BitSubsets.select items bits) : Int) ↔
      target.toNat ≤ familySize p (ProjectEulerSolutions.BitSubsets.select items bits) := by omega
  simp only [accepted, ProjectEulerSolutions.PrimeReplacementSearch.accepts, threshold, score]

/-- On an actual digit-position group, the source mask loop has the verified reference search result. -/
theorem group_result (p original posAddress powerAddress primeAddress : Nat) (target : Int) (positive : 0 < p)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getOriginal : env "d" = some (number original))
    (getPositions : env "pos" = some (.ref posAddress)) (getPowers : env "pow10" = some (.ref powerAddress))
    (getLength : env "m" = some (number (ProjectEulerSolutions.PrimeReplacementSearch.group target.toNat p original).length))
    (getPrimes : env "is_prime" = some (.ref primeAddress)) (getTarget : env "target" = some (number target))
    (positionsFound : heap[posAddress]? = some (.list (numbers (ProjectEulerSolutions.PrimeReplacementSearch.group target.toNat p original))))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers (Nat.digits 10 p).length))))
    (primesFound : heap[primeAddress]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Run p (Stable env) env [positionBody[3]'(by decide)] heap
      (ProjectEulerSolutions.PrimeReplacementSearch.query target.toNat p original) := by
  have scanned := result p original posAddress powerAddress primeAddress target
    (ProjectEulerSolutions.PrimeReplacementSearch.group target.toNat p original) positive
    (ProjectEulerSolutions.PrimeReplacementSearch.group_nodup _ _ _)
    (ProjectEulerSolutions.PrimeReplacementSearch.group_bounded _ _ _)
    (ProjectEulerSolutions.PrimeReplacementSearch.group_same _ _ _)
    env heap bytes getP getOriginal getPositions getPowers getLength getPrimes getTarget positionsFound powersFound primesFound size correct
  simpa only [ProjectEulerSolutions.PrimeReplacementSearch.query, accepted_nat] using scanned

end masks

namespace groups
open ProjectEulerSolutions
private abbrev refs (addresses : List Nat) : List Mutable.Value := addresses.map Mutable.Value.ref
private def writes : List String := ["d", "pos", "m"] ++ masks.writes
private def Stable (before after : HeapCalls.Env) : Prop := ∀ name, name ∉ writes → after name = before name
private theorem stable_refl (env : HeapCalls.Env) : Stable env env := fun _ _ => rfl
private theorem stable_trans (first : Stable before middle) (second : Stable middle after) : Stable before after :=
  fun name untouched => (second name untouched).trans (first name untouched)
private theorem stable_set (env : HeapCalls.Env) (name : String) (value : Mutable.Value) (written : name ∈ writes) :
    Stable env (setLocal env name value) := by
  intro key untouched
  have different : key ≠ name := by intro same; subst key; exact untouched written
  simp only [setLocal, different, if_false]
private theorem stable_masks (saved : masks.Stable before after) : Stable before after :=
  fun name untouched => saved name (fun member => untouched (List.mem_append_right _ member))

private structure Inputs (p parentAddress powerAddress primeAddress : Nat) (target : Int) (env : HeapCalls.Env) : Prop where
  getP : env "p" = some (number p)
  getGroups : env "pos_lists" = some (.ref parentAddress)
  getPowers : env "pow10" = some (.ref powerAddress)
  getPrimes : env "is_prime" = some (.ref primeAddress)
  getTarget : env "target" = some (number target)

private theorem Inputs.preserved (inputs : Inputs p parentAddress powerAddress primeAddress target before)
    (saved : Stable before after) : Inputs p parentAddress powerAddress primeAddress target after :=
  ⟨(saved "p" (by decide)).trans inputs.getP, (saved "pos_lists" (by decide)).trans inputs.getGroups,
    (saved "pow10" (by decide)).trans inputs.getPowers, (saved "is_prime" (by decide)).trans inputs.getPrimes,
    (saved "target" (by decide)).trans inputs.getTarget⟩

/-- The actual source comprehension creates ten independent empty rows and records their fresh addresses. -/
theorem allocate_rows (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockContinues program env [primeBody[3]'(by decide)] heap
      (setLocal env "pos_lists" (.ref heap.length)) (HeapRows.empty heap 10) := by
  exact BlockContinues.annAssign (HeapRows.allocate program "_" heap 10 env)

private def fillBody : List Stmt := match (primeBody[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []

/-- Appending the current position updates exactly its digit's row, matching the mathematical prefix groups. -/
theorem append_current (digits : List Nat) (base index digit : Nat) (digitBound : digit < 10)
    (found : digits[index]? = some digit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getParent : env "pos_lists" = some (.ref base)) (getDigit : env "dig" = some (number digit))
    (getIndex : env "i" = some (number index))
    (parentFound : heap[base]? = some (.list ((HeapRows.addresses base 10).map Mutable.Value.ref)))
    (rowsFound : ∀ original < 10, heap[base + 1 + original]? =
      some (.list (numbers (PrimeReplacementSearch.prefixGroup digits index original)))) :
    let finalHeap := heap.set (base + 1 + digit)
      (.list (numbers (PrimeReplacementSearch.prefixGroup digits index digit ++ [index])))
    BlockContinues program env fillBody heap env finalHeap ∧
      finalHeap[base]? = heap[base]? ∧
      (∀ original < 10, finalHeap[base + 1 + original]? =
        some (.list (numbers (PrimeReplacementSearch.prefixGroup digits (index + 1) original)))) := by
  dsimp only
  have receiver : Evaluates program env (.subscript (.name "pos_lists") (.name "dig")) heap (.ref (base + 1 + digit)) heap := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getParent) (Evaluates.lookup getDigit)
    simpa only [HeapRows.addresses, List.getElem_map, List.getElem_range] using
      Mutable.getItem_list heap base digit ((HeapRows.addresses base 10).map Mutable.Value.ref) parentFound
        (by simpa only [HeapRows.addresses, List.length_map, List.length_range] using digitBound)
  have row := rowsFound digit digitBound
  refine ⟨?_, ?_, ?_⟩
  · have appended := BlockContinues.discard (Evaluates.append receiver row (Evaluates.lookup getIndex) row)
    simpa only [numbers, List.map_append, List.map_singleton] using appended
  · exact List.getElem?_set_ne (by omega)
  · intro original inside
    have step := PrimeReplacementSearch.prefixGroup_step digits index original digit found
    by_cases same : digit = original
    · subst original
      have live := (List.getElem?_eq_some_iff.mp row).1
      simpa only [step, if_pos rfl] using List.getElem?_set_self live
    · rw [List.getElem?_set_ne (by omega)]
      simpa only [step, if_neg same, List.append_nil] using rowsFound original inside

private def FillStable (before after : HeapCalls.Env) : Prop :=
  ∀ name, name ≠ "i" → name ≠ "dig" → after name = before name

private def RowsFrame (base : Nat) (before after : Mutable.Heap) : Prop :=
  ∀ address, (∀ digit < 10, address ≠ base + 1 + digit) → after[address]? = before[address]?

/-- The enumeration fills every digit row while leaving the source sequence and all other heap cells intact. -/
theorem fill_loop (digits : List Nat) (base sourceAddress count first remaining : Nat)
    (within : count ≤ digits.length) (endpoint : first + remaining = count)
    (digitBounds : ∀ digit ∈ digits, digit < 10)
    (sourceOutside : ∀ digit < 10, sourceAddress ≠ base + 1 + digit)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (getParent : env "pos_lists" = some (.ref base))
    (parentFound : heap[base]? = some (.list ((HeapRows.addresses base 10).map Mutable.Value.ref)))
    (sourceFound : heap[sourceAddress]? = some (.list (numbers (digits.take count))))
    (rowsFound : ∀ original < 10, heap[base + 1 + original]? =
      some (.list (numbers (PrimeReplacementSearch.prefixGroup digits first original)))) :
    ∃ finalEnv finalHeap,
      BlockContinues program env [.enumerateStepLoop "i" "dig" (.ref sourceAddress) first first fillBody []] heap finalEnv finalHeap ∧
      finalHeap[base]? = some (.list ((HeapRows.addresses base 10).map Mutable.Value.ref)) ∧
      (∀ original < 10, finalHeap[base + 1 + original]? =
        some (.list (numbers (PrimeReplacementSearch.prefixGroup digits count original)))) ∧
      finalHeap[sourceAddress]? = some (.list (numbers (digits.take count))) ∧
      FillStable env finalEnv ∧ RowsFrame base heap finalHeap ∧ finalHeap.length = heap.length := by
  induction remaining generalizing first env heap with
  | zero =>
      refine ⟨env, heap, BlockContinues.enumerate_done (source := .ref sourceAddress) (items := numbers (digits.take count))
        (by simp [sequenceValue, Mutable.sequence, sourceFound]) ?_ .nil, parentFound, ?_, sourceFound,
        (fun _ _ _ => rfl), (fun _ _ => rfl), rfl⟩
      · exact List.getElem?_eq_none (by simp only [numbers, List.length_map, List.length_take]; omega)
      · simpa only [show first = count by omega] using rowsFound
  | succ remaining ih =>
      have inside : first < digits.length := by omega
      have chosen : first < count := by omega
      let digit := digits[first]
      have digitBound : digit < 10 := digitBounds digit (List.getElem_mem inside)
      have read : digits[first]? = some digit := List.getElem?_eq_getElem inside
      let bound := setLocal (setLocal env "i" (number first)) "dig" (number digit)
      let nextHeap := heap.set (base + 1 + digit)
        (.list (numbers (PrimeReplacementSearch.prefixGroup digits first digit ++ [first])))
      obtain ⟨appended, parentSaved, nextRows⟩ := append_current digits base first digit digitBound read bound heap
        (by simpa [bound, setLocal] using getParent) rfl (by simp [bound, setLocal]) parentFound rowsFound
      have nextParent : nextHeap[base]? = some (.list ((HeapRows.addresses base 10).map Mutable.Value.ref)) :=
        parentSaved.trans parentFound
      have nextSource : nextHeap[sourceAddress]? = some (.list (numbers (digits.take count))) := by
        rw [List.getElem?_set_ne (Ne.symm (sourceOutside digit digitBound))]
        exact sourceFound
      obtain ⟨finalEnv, finalHeap, executed, finalParent, finalRows, finalSource, preserved, framed, sameLength⟩ :=
        ih (first+1) (by omega) bound nextHeap (by simpa [bound, setLocal] using getParent) nextParent nextSource nextRows
      refine ⟨finalEnv, finalHeap, ?_, finalParent, finalRows, finalSource, ?_, ?_, ?_⟩
      · apply BlockContinues.enumerate_step (source := .ref sourceAddress) (items := numbers (digits.take count))
          (by simp [sequenceValue, Mutable.sequence, sourceFound])
          (value := number digit)
        · simp only [numbers, List.getElem?_map, List.getElem?_take_of_lt chosen, read, Option.map_some]
        · exact appended.append executed
      · intro name hi hd
        exact (preserved name hi hd).trans (by simp only [bound, setLocal, hi, hd, if_false])
      · intro address outside
        exact (framed address outside).trans (List.getElem?_set_ne (Ne.symm (outside digit digitBound)))
      · simpa only [nextHeap, List.length_set] using sameLength

private def digitSource : Expr := match (primeBody[4]? : Option Stmt) with
  | some (.forIn _ (.call _ [source]) _ _) => source | _ => .constant .none
private def digitCount (target : Int) (digits : List Nat) : Nat :=
  if 8 ≤ target then digits.length - 1 else digits.length

/-- The source's conditional slice drops only the units position for an eight-prime search. -/
theorem select_digits (digits : List Nat) (target : Int) (address : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) (getDigits : env "ds" = some (.ref address))
    (found : heap[address]? = some (.list (numbers digits))) :
    ∃ selectedAddress extra, Evaluates program env digitSource heap (.ref selectedAddress) (heap ++ extra) ∧
      (heap ++ extra)[selectedAddress]? = some (.list (numbers (digits.take (digitCount target digits)))) ∧
      (selectedAddress = address ∨ selectedAddress = heap.length) := by
  have test : Evaluates program env (.compare (.name "target") [(.gtE, .constant (.int 8))]) heap
      (flag (decide (8 ≤ target))) heap := by
    apply Evaluates.compare (av := number target) (bv := number 8) (middle := heap) (afterRight := heap)
      (Evaluates.lookup getTarget) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases large : 8 ≤ target
  · have lengthEq : FormalSW.Py.Builtin.Slice.prefixLength (numbers digits).length (-1) = digits.length - 1 := by
      simp only [numbers, List.length_map, FormalSW.Py.Builtin.Slice.prefixLength]
      omega
    have sliced : Evaluates program env
        (.subscript (.name "ds") (.slice none (some (.unaryOp .uSub (.constant (.int 1)))) none))
        heap (.ref heap.length) (heap ++ [.list (numbers (digits.take (digits.length-1)))]) := by
      apply Evaluates.slice (receiver := .ref address) (first := .scalar .none) (last := number (-1)) (stepValue := .scalar .none) (middle := heap)
        (.cons (Evaluates.lookup getDigits) (.cons Evaluates.constant (.cons (Evaluates.unary Evaluates.constant rfl) (.cons Evaluates.constant .nil))))
      have operation := Mutable.getSlice_list_prefix heap address (numbers digits) (-1) found
      rw [lengthEq] at operation
      simpa only [numbers, List.map_take] using operation
    refine ⟨heap.length, [.list (numbers (digits.take (digitCount target digits)))], ?_, by simp, Or.inr rfl⟩
    apply Evaluates.conditional (flag := true) (value := flag true)
      (by simpa only [large, decide_true] using test) (by simp [Mutable.truth, scalarTruth])
    simpa only [digitCount, large, if_true] using sliced
  · refine ⟨address, [], ?_, ?_, Or.inl rfl⟩
    · simp only [List.append_nil]
      apply Evaluates.conditional (flag := false) (value := flag false)
        (by simpa only [large, decide_false] using test) (by simp [Mutable.truth, scalarTruth])
      exact Evaluates.lookup getDigits
    · simpa only [List.append_nil, digitCount, large, if_false, List.take_length] using found

/-- The source allocation, conditional slice, and grouping loop construct every reference digit-position group. -/
theorem build (p : Nat) (target : Int) (digitsAddress : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) (getDigits : env "ds" = some (.ref digitsAddress))
    (found : heap[digitsAddress]? = some (.list (numbers (Nat.digits 10 p).reverse))) :
    ∃ finalEnv finalHeap,
      BlockContinues program env ((primeBody.drop 3).take 2) heap finalEnv finalHeap ∧
      finalEnv "pos_lists" = some (.ref heap.length) ∧
      finalHeap[heap.length]? = some (.list ((HeapRows.addresses heap.length 10).map Mutable.Value.ref)) ∧
      (∀ original < 10, finalHeap[heap.length + 1 + original]? =
        some (.list (numbers (PrimeReplacementSearch.group target.toNat p original)))) ∧
      (∀ (address : Nat) (object : Mutable.Object), heap[address]? = some object → finalHeap[address]? = some object) ∧
      (∀ name, name ∉ ["pos_lists", "i", "dig"] → finalEnv name = env name) := by
  let digits := (Nat.digits 10 p).reverse
  let initialized := HeapRows.empty heap 10
  let ready := setLocal env "pos_lists" (.ref heap.length)
  have allocated := allocate_rows env heap
  have savedDigits := HeapRows.prefix_found heap 10 digitsAddress (.list (numbers digits)) found
  obtain ⟨selectedAddress, extra, selected, chosenFound, location⟩ := select_digits digits target digitsAddress ready initialized
    (by simpa [ready, setLocal] using getTarget) (by simpa [ready, setLocal] using getDigits) savedDigits
  have parent := HeapCollection.prefix_found initialized extra heap.length _ (HeapRows.parent_found heap 10)
  have rows : ∀ original < 10, (initialized ++ extra)[heap.length + 1 + original]? =
      some (.list (numbers (PrimeReplacementSearch.prefixGroup digits 0 original))) := by
    intro original inside
    exact HeapCollection.prefix_found initialized extra (heap.length + 1 + original) _ (HeapRows.row_found heap 10 original inside)
  have countWithin : digitCount target digits ≤ digits.length := by unfold digitCount; split_ifs <;> omega
  have digitBounds : ∀ digit ∈ digits, digit < 10 := by
    intro digit member
    exact Nat.digits_lt_base (by decide : 1 < 10) (by simpa only [digits, List.mem_reverse] using member)
  have sourceOutside : ∀ digit < 10, selectedAddress ≠ heap.length + 1 + digit := by
    intro digit inside equal
    rcases location with old | fresh
    · have live := (List.getElem?_eq_some_iff.mp found).1
      rw [old] at equal
      omega
    · have live : heap.length + 1 + digit < initialized.length :=
        (List.getElem?_eq_some_iff.mp (HeapRows.row_found heap 10 digit inside)).1
      exact (Nat.ne_of_lt live) (equal.symm.trans fresh)
  obtain ⟨finalEnv, finalHeap, executed, finalParent, finalRows, _, preserved, framed, _⟩ :=
    fill_loop digits heap.length selectedAddress (digitCount target digits) 0 (digitCount target digits) countWithin (by omega)
      digitBounds sourceOutside ready (initialized ++ extra) rfl parent chosenFound rows
  have filled : BlockContinues program ready [primeBody[4]'(by decide)] initialized finalEnv finalHeap := by
    apply BlockContinues.for_enumerate (source := .ref selectedAddress) (items := numbers (digits.take (digitCount target digits)))
      selected (by simp [Mutable.sequence, chosenFound]) executed
  refine ⟨finalEnv, finalHeap, allocated.append filled, (preserved "pos_lists" (by decide) (by decide)).trans rfl,
    finalParent, ?_, ?_, ?_⟩
  · intro original inside
    have same : PrimeReplacementSearch.prefixGroup digits (digitCount target digits) original =
        PrimeReplacementSearch.group target.toNat p original := by
      rw [PrimeReplacementSearch.group_eq_prefix]
      have threshold : 8 ≤ target ↔ 8 ≤ target.toNat := by omega
      simp only [digits, digitCount, List.length_reverse, threshold]
    simpa only [same] using finalRows original inside
  · intro address object previous
    have live := (List.getElem?_eq_some_iff.mp previous).1
    exact (framed address (fun _ _ => by omega)).trans
      (HeapCollection.prefix_found initialized extra address object (HeapRows.prefix_found heap 10 address object previous))
  · intro name untouched
    have hi : name ≠ "i" := fun equal => untouched (by simp [equal])
    have hd : name ≠ "dig" := fun equal => untouched (by simp [equal])
    have hp : name ≠ "pos_lists" := fun equal => untouched (by simp [equal])
    exact (preserved name hi hd).trans (by simp only [ready, setLocal, hp, if_false])

private def emptyTest : Expr := match (positionBody[1]? : Option Stmt) with
  | some (.ifThen condition _ _) => condition | _ => .constant .none
private def smallTest : Expr := match (positionBody[2]? : Option Stmt) with
  | some (.ifThen condition _ _) => condition | _ => .constant .none

private theorem empty_evaluates (length : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLength : env "m" = some (number length)) :
    Evaluates program env emptyTest heap (flag (decide (length = 0))) heap := by
  apply Evaluates.compare (av := number length) (bv := number 0) (middle := heap) (afterRight := heap)
    (Evaluates.lookup getLength) Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

private theorem small_evaluates (target : Int) (length : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) (getLength : env "m" = some (number length)) :
    Evaluates program env smallTest heap (flag (decide (8 ≤ target ∧ length < 3))) heap := by
  have threshold : Evaluates program env (.compare (.name "target") [(.gtE, .constant (.int 8))]) heap
      (flag (decide (8 ≤ target))) heap := by
    apply Evaluates.compare (av := number target) (bv := number 8) (middle := heap) (afterRight := heap)
      (Evaluates.lookup getTarget) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases large : 8 ≤ target
  · apply Evaluates.boolean_next (value := flag true) (flag := true)
      (by simpa only [large, decide_true] using threshold) (by simp [Mutable.truth, scalarTruth]) (by decide)
    apply Evaluates.boolean_last (flag := decide (length < 3))
    · apply Evaluates.compare (av := number length) (bv := number 3) (middle := heap) (afterRight := heap)
        (Evaluates.lookup getLength) Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, large]
    · simp [Mutable.truth, scalarTruth, large]
  · simpa [smallTest, large] using Evaluates.boolean_stop (op := BoolOp.and)
      (rest := [.compare (.name "m") [(.lt, .constant (.int 3))]])
      (value := flag false) (flag := false) (by simpa only [large, decide_false] using threshold)
      (by simp [Mutable.truth, scalarTruth]) (by decide)

private theorem run_enumerate_step
    (sequence : Mutable.sequence (sequenceValue source) heap = some (.ok items, heap))
    (item : items[position]? = some value)
    (executed : masks.Run p invariant (setLocal (setLocal env index (number counter)) name value)
      (body ++ [Calls.loopContinuation body (.enumerateStepLoop index name source (position+1) (counter+1) body [])]) heap found) :
    masks.Run p invariant env [.enumerateStepLoop index name source position counter body []] heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, continued, valid⟩ := executed
      exact ⟨finalEnv, BlockContinues.enumerate_step sequence item continued, valid⟩
  | true =>
      intro following
      apply BlockReturns.enumerate_step sequence item
      simpa only [List.append_assoc] using executed following

private theorem loop (p parentAddress powerAddress primeAddress first remaining : Nat) (target : Int)
    (addresses : List Nat) (length : addresses.length = 10) (positive : 0 < p) (endpoint : first + remaining = 10)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (inputs : Inputs p parentAddress powerAddress primeAddress target env)
    (parentFound : heap[parentAddress]? = some (.list (refs addresses)))
    (rowsFound : ∀ digit (inside : digit < addresses.length), heap[addresses[digit]]? =
      some (.list (numbers (PrimeReplacementSearch.group target.toNat p digit))))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers (Nat.digits 10 p).length))))
    (primesFound : heap[primeAddress]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    masks.Run p (fun finalEnv => Inputs p parentAddress powerAddress primeAddress target finalEnv ∧ Stable env finalEnv)
      env [.enumerateStepLoop "d" "pos" (.ref parentAddress) first first positionBody []] heap
      ((List.range' first remaining).any (PrimeReplacementSearch.query target.toNat p)) := by
  have sequence : Mutable.sequence (.ref parentAddress) heap = some (.ok (refs addresses), heap) := by
    simp [Mutable.sequence, parentFound]
  induction remaining generalizing first env with
  | zero =>
      refine ⟨env, BlockContinues.enumerate_done (source := .ref parentAddress) sequence ?_ .nil, inputs, stable_refl env⟩
      simp [refs, show first = addresses.length by omega]
  | succ remaining ih =>
      have inside : first < addresses.length := by omega
      let items := PrimeReplacementSearch.group target.toNat p first
      let bound := setLocal (setLocal env "d" (number first)) "pos" (.ref addresses[first])
      let counted := setLocal bound "m" (number items.length)
      let next := Stmt.enumerateStepLoop "d" "pos" (.ref parentAddress) (first+1) (first+1 : Nat) positionBody []
      have prefixStable : Stable env counted := stable_trans (stable_set env "d" _ (by decide))
        (stable_trans (stable_set (setLocal env "d" (number first)) "pos" _ (by decide)) (stable_set bound "m" _ (by decide)))
      have countedInputs := inputs.preserved prefixStable
      have readLength : BlockContinues program bound [positionBody[0]'(by decide)] heap counted heap := by
        apply BlockContinues.assign
        apply Evaluates.builtin (values := [.ref addresses[first]]) (middle := heap)
          (.cons (Evaluates.lookup (by simp [bound, setLocal])) .nil)
        simp [primitive, Mutable.length, rowsFound first inside, numbers, items]
      have emptyTested := empty_evaluates items.length counted heap rfl
      have smallTested := small_evaluates target items.length counted heap countedInputs.getTarget rfl
      have current : (refs addresses)[first]? = some (.ref addresses[first]) := by
        simp only [List.getElem?_map, List.getElem?_eq_getElem inside, Option.map_some]
      by_cases empty : items.length = 0
      · have rejected : PrimeReplacementSearch.query target.toNat p first = false := by
          simp only [PrimeReplacementSearch.query, ← show items = PrimeReplacementSearch.group target.toNat p first from rfl,
            empty, Nat.pow_zero, Nat.sub_self, List.range'_zero, List.any_nil]
        have inherited := masks.run_weaken
          (invariant := fun finalEnv => Inputs p parentAddress powerAddress primeAddress target finalEnv ∧ Stable counted finalEnv)
          (weaker := fun finalEnv => Inputs p parentAddress powerAddress primeAddress target finalEnv ∧ Stable env finalEnv)
          (fun finalEnv valid => ⟨valid.1, stable_trans prefixStable valid.2⟩)
          (ih (first+1) (by omega) counted countedInputs)
        rw [List.range'_succ, List.any_cons, rejected, Bool.false_or]
        apply run_enumerate_step (source := .ref parentAddress) sequence current
        change masks.Run p _ bound ([positionBody[0]'(by decide)] ++ (positionBody.drop 1 ++ [.loopResume next])) heap _
        apply masks.run_prepend readLength
        apply masks.run_branch (flag := true) (value := flag true)
          (by simpa only [empty, decide_true] using emptyTested) (by simp [Mutable.truth, scalarTruth])
        exact masks.run_continue (next := next) (rest := []) rfl inherited
      · by_cases small : 8 ≤ target ∧ items.length < 3
        · have rejected : PrimeReplacementSearch.query target.toNat p first = false :=
            PrimeReplacementSearch.query_false_of_small _ _ _ (by omega) small.2
          have inherited := masks.run_weaken
            (invariant := fun finalEnv => Inputs p parentAddress powerAddress primeAddress target finalEnv ∧ Stable counted finalEnv)
            (weaker := fun finalEnv => Inputs p parentAddress powerAddress primeAddress target finalEnv ∧ Stable env finalEnv)
            (fun finalEnv valid => ⟨valid.1, stable_trans prefixStable valid.2⟩)
            (ih (first+1) (by omega) counted countedInputs)
          rw [List.range'_succ, List.any_cons, rejected, Bool.false_or]
          apply run_enumerate_step (source := .ref parentAddress) sequence current
          change masks.Run p _ bound ([positionBody[0]'(by decide)] ++ (positionBody.drop 1 ++ [.loopResume next])) heap _
          apply masks.run_prepend readLength
          apply masks.run_branch (flag := false) (value := flag false)
            (by simpa only [empty, decide_false] using emptyTested) (by simp [Mutable.truth, scalarTruth])
          change masks.Run p _ counted (positionBody.drop 2 ++ [.loopResume next]) heap _
          apply masks.run_branch (flag := true) (value := flag true)
            (by simpa only [show decide (8 ≤ target ∧ items.length < 3) = true from decide_eq_true small] using smallTested)
            (by simp [Mutable.truth, scalarTruth])
          exact masks.run_continue (next := next) (rest := []) rfl inherited
        · have scanned := masks.group_result p first addresses[first] powerAddress primeAddress target positive counted heap bytes
            countedInputs.getP (by simp [counted, bound, setLocal]) (by simp [counted, bound, setLocal]) countedInputs.getPowers rfl
            countedInputs.getPrimes countedInputs.getTarget (rowsFound first inside) powersFound primesFound size correct
          cases found : PrimeReplacementSearch.query target.toNat p first with
          | true =>
              rw [found] at scanned
              rw [List.range'_succ, List.any_cons, found, Bool.true_or]
              apply run_enumerate_step (source := .ref parentAddress) sequence current
              change masks.Run p _ bound ([positionBody[0]'(by decide)] ++ (positionBody.drop 1 ++ [.loopResume next])) heap true
              apply masks.run_prepend readLength
              apply masks.run_branch (flag := false) (value := flag false)
                (by simpa only [empty, decide_false] using emptyTested) (by simp [Mutable.truth, scalarTruth])
              change masks.Run p _ counted (positionBody.drop 2 ++ [.loopResume next]) heap true
              apply masks.run_branch (flag := false) (value := flag false)
                (by simpa only [small, decide_false] using smallTested) (by simp [Mutable.truth, scalarTruth])
              intro following
              simpa only [List.append_assoc] using scanned ([.loopResume next] ++ following)
          | false =>
              rw [found] at scanned
              obtain ⟨middleEnv, continued, saved⟩ := scanned
              have combined := stable_trans prefixStable (stable_masks saved)
              have middleInputs := inputs.preserved combined
              have inherited := masks.run_weaken
                (invariant := fun finalEnv => Inputs p parentAddress powerAddress primeAddress target finalEnv ∧ Stable middleEnv finalEnv)
                (weaker := fun finalEnv => Inputs p parentAddress powerAddress primeAddress target finalEnv ∧ Stable env finalEnv)
                (fun finalEnv valid => ⟨valid.1, stable_trans combined valid.2⟩)
                (ih (first+1) (by omega) middleEnv middleInputs)
              rw [List.range'_succ, List.any_cons, found, Bool.false_or]
              apply run_enumerate_step (source := .ref parentAddress) sequence current
              change masks.Run p _ bound ([positionBody[0]'(by decide)] ++ (positionBody.drop 1 ++ [.loopResume next])) heap _
              apply masks.run_prepend readLength
              apply masks.run_branch (flag := false) (value := flag false)
                (by simpa only [empty, decide_false] using emptyTested) (by simp [Mutable.truth, scalarTruth])
              change masks.Run p _ counted (positionBody.drop 2 ++ [.loopResume next]) heap _
              apply masks.run_branch (flag := false) (value := flag false)
                (by simpa only [small, decide_false] using smallTested) (by simp [Mutable.truth, scalarTruth])
              change masks.Run p _ counted ([positionBody[3]'(by decide)] ++ [.loopResume next]) heap _
              exact masks.run_prepend continued (masks.run_resume inherited)

private theorem run_for_enumerate (iterable : Evaluates program env expression heap source heap)
    (sequence : Mutable.sequence source heap = some (.ok items, heap))
    (executed : masks.Run p invariant env [.enumerateStepLoop index name (sequenceSource source) 0 0 body []] heap found) :
    masks.Run p invariant env [.forIn (.tuple [.name index, .name name]) (.call (.name "enumerate") [expression]) body []] heap found := by
  cases found with
  | false =>
      obtain ⟨finalEnv, continued, valid⟩ := executed
      exact ⟨finalEnv, BlockContinues.for_enumerate iterable sequence continued, valid⟩
  | true =>
      intro following
      exact BlockReturns.for_enumerate iterable sequence (executed following)

/-- Searching all ten source digit groups returns precisely when the prime has a qualifying reference family. -/
theorem result (p parentAddress powerAddress primeAddress : Nat) (target : Int) (prime : Nat.Prime p)
    (addresses : List Nat) (length : addresses.length = 10) (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getGroups : env "pos_lists" = some (.ref parentAddress))
    (getPowers : env "pow10" = some (.ref powerAddress)) (getPrimes : env "is_prime" = some (.ref primeAddress))
    (getTarget : env "target" = some (number target))
    (parentFound : heap[parentAddress]? = some (.list (refs addresses)))
    (rowsFound : ∀ digit (inside : digit < addresses.length), heap[addresses[digit]]? =
      some (.list (numbers (PrimeReplacementSearch.group target.toNat p digit))))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers (Nat.digits 10 p).length))))
    (primesFound : heap[primeAddress]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    masks.Run p (Stable env) env [primeBody[5]'(by decide)] heap (ProjectEulerStatements.P51.hasPrimeFamily target.toNat p) := by
  have sequence : Mutable.sequence (.ref parentAddress) heap = some (.ok (refs addresses), heap) := by
    simp [Mutable.sequence, parentFound]
  have scanned := loop p parentAddress powerAddress primeAddress 0 10 target addresses length prime.pos rfl env heap bytes
    ⟨getP, getGroups, getPowers, getPrimes, getTarget⟩ parentFound rowsFound powersFound primesFound size correct
  have searched := run_for_enumerate (Evaluates.lookup getGroups) sequence scanned
  have same : (List.range 10).any (PrimeReplacementSearch.query target.toNat p) = ProjectEulerStatements.P51.hasPrimeFamily target.toNat p := by
    apply Bool.eq_iff_iff.mpr
    simpa only [prime, true_and] using (PrimeReplacementSearch.hasPrimeFamily_iff_query target.toNat p).symm
  simpa only [← List.range_eq_range', same] using masks.run_weaken (weaker := Stable env) (fun _ valid => valid.2) searched

end groups

namespace decimalDigits
private abbrev text (digits : List Nat) := String.ofList (digits.map Nat.digitChar)
private abbrev character (digit : Nat) : Mutable.Value := .scalar (.string (String.singleton (Nat.digitChar digit)))
private def expression : Expr := .binOp (.builtinCall .ord [.name "c"]) .sub (.constant (.int 48))
private def body (address : Nat) : List Stmt := [.comprehensionAppend address expression]

private theorem ordinal_value (digit : Nat) (decimal : digit < 10) :
    ((Nat.digitChar digit).toNat : Int) - 48 = (digit : Int) := by
  interval_cases digit <;> decide +kernel

private theorem ordinal (digit : Nat) (decimal : digit < 10) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getCharacter : env "c" = some (character digit)) : Evaluates program env expression heap (number digit) heap := by
  apply Evaluates.binary (av := number (Nat.digitChar digit).toNat) (bv := number 48) (middle := heap) (afterRight := heap)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getCharacter) .nil)
    change scalarResult (some (Strings.ord (.string (String.ofList [Nat.digitChar digit])))) heap = _
    rw [Strings.ord_character]
    rfl
  · exact Evaluates.constant
  · simp [Mutable.binary, scalarBinary, ordinal_value digit decimal]

private theorem loop (items remaining : List Nat) (position : Nat) (base : Mutable.Heap) (before : List Nat)
    (env : HeapCalls.Env) (suffix : items.drop position = remaining) (decimal : ∀ digit ∈ remaining, digit < 10) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "c" (.scalar (.string (text items))) position (body base.length) []]
      (ProjectEulerSolutions.HeapCollection.working base (numbers before) []) finalEnv
      (ProjectEulerSolutions.HeapCollection.working base (numbers (before ++ remaining)) []) := by
  have sequence (stored : List Nat) : Mutable.sequence (.scalar (.string (text items)))
      (ProjectEulerSolutions.HeapCollection.working base (numbers stored) []) =
      some (.ok (items.map character), ProjectEulerSolutions.HeapCollection.working base (numbers stored) []) := by
    simp [Mutable.sequence, text, character, List.map_map]
  induction remaining generalizing position before env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.append_nil]
      apply BlockContinues.sequence_done (source := .scalar (.string (text items))) (sequence before)
      · have missing : items[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [missing]
      · exact .nil
  | cons head tail ih =>
      have item : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      let bound := setLocal env "c" (character head)
      have evaluated := ordinal head (decimal head (by simp)) bound
        (ProjectEulerSolutions.HeapCollection.working base (numbers before) []) rfl
      have emitted := ProjectEulerSolutions.HeapCollection.emit base (numbers before) [] [] (number head)
        (by simpa only [List.append_nil] using evaluated)
      have emitted' : BlockContinues program bound (body base.length)
          (ProjectEulerSolutions.HeapCollection.working base (numbers before) []) bound
          (ProjectEulerSolutions.HeapCollection.working base (numbers (before ++ [head])) []) := by
        simpa only [body, numbers, List.map_append, List.map_singleton, List.nil_append] using emitted
      have suffixTail : items.drop (position+1) = tail := by simp [List.drop_add_one_eq_tail_drop, suffix]
      obtain ⟨finalEnv, executed⟩ := ih (position+1) (before ++ [head]) bound suffixTail
        (fun digit member => decimal digit (by simp [member]))
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (source := .scalar (.string (text items))) (value := character head) (sequence before) (by simp [item])
      simpa only [body, List.append_assoc, List.singleton_append] using emitted'.append executed

/-- The actual character comprehension produces the canonical most-significant-first decimal digits. -/
theorem result (p : Nat) (positive : 0 < p) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getString : env "s" = some (.scalar (.string (toString (p : Int))))) :
    Evaluates program env (.listComp expression [("c", .name "s", [])]) heap (.ref heap.length)
      (heap ++ [.list (numbers (Nat.digits 10 p).reverse)]) := by
  let digits := (Nat.digits 10 p).reverse
  have equal : toString (p : Int) = text digits := by
    apply String.toList_injective
    simpa [toString, Int.repr, Nat.repr, text, digits] using ProjectEulerStatements.DecimalDigits.toDigits_eq_digitChars p positive
  have decimal : ∀ digit ∈ digits, digit < 10 := by
    intro digit member
    exact Nat.digits_lt_base (by decide : 1 < 10) (by simpa only [digits, List.mem_reverse] using member)
  obtain ⟨finalEnv, executed⟩ := loop digits digits 0 heap [] (comprehensionEnv env [("c", .name "s", [])]) rfl decimal
  apply Evaluates.list_comp_sequence (value := .scalar (.string (text digits))) (items := digits.map character)
    (by intros; intro impossible; cases impossible)
    (Evaluates.lookup (by simpa only [equal] using getString))
    (by simp [Mutable.sequence, text, character, List.map_map])
  exact executed

/-- The two source statements build the decimal string and its fresh digit list. -/
theorem prepare (p : Nat) (positive : 0 < p) (env : HeapCalls.Env) (heap : Mutable.Heap) (getP : env "p" = some (number p)) :
    BlockContinues program env ((primeBody.drop 1).take 2) heap
      (setLocal (setLocal env "s" (.scalar (.string (toString (p : Int))))) "ds" (.ref heap.length))
      (heap ++ [.list (numbers (Nat.digits 10 p).reverse)]) := by
  let stringEnv := setLocal env "s" (.scalar (.string (toString (p : Int))))
  have first : BlockContinues program env [primeBody[1]'(by decide)] heap stringEnv heap :=
    BlockContinues.assign (Evaluates.builtin (.cons (Evaluates.lookup getP) .nil) rfl)
  have second : BlockContinues program stringEnv [primeBody[2]'(by decide)] heap
      (setLocal stringEnv "ds" (.ref heap.length)) (heap ++ [.list (numbers (Nat.digits 10 p).reverse)]) :=
    BlockContinues.assign (result p positive stringEnv heap rfl)
  exact first.append second

end decimalDigits

namespace candidate
open ProjectEulerSolutions
private def writes : List String := ["s", "ds", "pos_lists", "i", "dig"] ++ groups.writes
private def Stable (before after : HeapCalls.Env) : Prop := ∀ name, name ∉ writes → after name = before name

/-- The complete source test for one prime constructs its groups and returns exactly for the reference family predicate. -/
theorem result (p powerAddress primeAddress : Nat) (target : Int) (prime : Nat.Prime p)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getP : env "p" = some (number p)) (getTarget : env "target" = some (number target))
    (getPowers : env "pow10" = some (.ref powerAddress)) (getPrimes : env "is_prime" = some (.ref primeAddress))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers (Nat.digits 10 p).length))))
    (primesFound : heap[primeAddress]? = some (.bytearray bytes)) (size : bytes.length = 10^(Nat.digits 10 p).length)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    ∃ finalHeap,
      (ProjectEulerStatements.P51.hasPrimeFamily target.toNat p = true → ∀ following,
        BlockReturns program env (primeBody.drop 1 ++ following) heap (number p) finalHeap) ∧
      (ProjectEulerStatements.P51.hasPrimeFamily target.toNat p = false → ∃ finalEnv,
        BlockContinues program env (primeBody.drop 1) heap finalEnv finalHeap ∧ Stable env finalEnv) ∧
      (∀ (address : Nat) (object : Mutable.Object), heap[address]? = some object → finalHeap[address]? = some object) := by
  let digitHeap := heap ++ [.list (numbers (Nat.digits 10 p).reverse)]
  let digitEnv := setLocal (setLocal env "s" (.scalar (.string (toString (p : Int))))) "ds" (.ref heap.length)
  have prepared := decimalDigits.prepare p prime.pos env heap getP
  obtain ⟨groupEnv, groupHeap, built, getGroups, parentFound, rowsFound, savedHeap, savedEnv⟩ :=
    groups.build p target heap.length digitEnv digitHeap (by simpa [digitEnv, setLocal] using getTarget) rfl (by simp [digitHeap])
  have outerHeap : ∀ (address : Nat) (object : Mutable.Object), heap[address]? = some object → groupHeap[address]? = some object := by
    intro address object previous
    exact savedHeap address object (HeapCollection.prefix_found heap _ address object previous)
  have readyP := (savedEnv "p" (by decide)).trans (by simpa [digitEnv, setLocal] using getP)
  have readyTarget := (savedEnv "target" (by decide)).trans (by simpa [digitEnv, setLocal] using getTarget)
  have readyPowers := (savedEnv "pow10" (by decide)).trans (by simpa [digitEnv, setLocal] using getPowers)
  have readyPrimes := (savedEnv "is_prime" (by decide)).trans (by simpa [digitEnv, setLocal] using getPrimes)
  have addressesLength : (HeapRows.addresses digitHeap.length 10).length = 10 := by simp [HeapRows.addresses]
  have rows : ∀ original (inside : original < (HeapRows.addresses digitHeap.length 10).length),
      groupHeap[(HeapRows.addresses digitHeap.length 10)[original]]? =
        some (.list (numbers (PrimeReplacementSearch.group target.toNat p original))) := by
    intro original inside
    simpa only [HeapRows.addresses, List.getElem_map, List.getElem_range] using rowsFound original (by simpa only [addressesLength] using inside)
  have searched := groups.result p digitHeap.length powerAddress primeAddress target prime (HeapRows.addresses digitHeap.length 10)
    addressesLength groupEnv groupHeap bytes readyP getGroups readyPowers readyPrimes readyTarget parentFound rows
    (outerHeap powerAddress _ powersFound) (outerHeap primeAddress _ primesFound) size correct
  have preparedBlock := prepared.append built
  refine ⟨groupHeap, ?_, ?_, outerHeap⟩
  · intro family following
    have returned := searched
    rw [family] at returned
    exact preparedBlock.returns (returned following)
  · intro noFamily
    have exhausted := searched
    rw [noFamily] at exhausted
    obtain ⟨finalEnv, continued, preserved⟩ := exhausted
    refine ⟨finalEnv, preparedBlock.append continued, ?_⟩
    intro name untouched
    have searchUntouched : name ∉ groups.writes := fun member => untouched (List.mem_append_right _ member)
    have noS : name ≠ "s" := fun equal => untouched (by simp [writes, equal])
    have noDs : name ≠ "ds" := fun equal => untouched (by simp [writes, equal])
    have noParent : name ≠ "pos_lists" := fun equal => untouched (by simp [writes, equal])
    have noI : name ≠ "i" := fun equal => untouched (by simp [writes, equal])
    have noDig : name ≠ "dig" := fun equal => untouched (by simp [writes, equal])
    exact (preserved name searchUntouched).trans ((savedEnv name (by simp [noParent, noI, noDig])).trans
      (by simp only [digitEnv, setLocal, noS, noDs, if_false]))

end candidate

namespace primeSearch
private def writes : List String := "p" :: candidate.writes
private def Stable (before after : HeapCalls.Env) : Prop := ∀ name, name ∉ writes → after name = before name
private def HeapFrame (before after : Mutable.Heap) : Prop :=
  ∀ (address : Nat) (object : Mutable.Object), before[address]? = some object → after[address]? = some object

private theorem stable_refl (env : HeapCalls.Env) : Stable env env := fun _ _ => rfl
private theorem stable_trans (first : Stable before middle) (second : Stable middle after) : Stable before after :=
  fun name untouched => (second name untouched).trans (first name untouched)
private theorem stable_set (env : HeapCalls.Env) (value : Mutable.Value) : Stable env (setLocal env "p" value) := by
  intro name untouched
  have different : name ≠ "p" := fun equal => untouched (by simp [writes, equal])
  simp only [setLocal, different, if_false]
private theorem stable_candidate (saved : candidate.Stable before after) : Stable before after :=
  fun name untouched => saved name (fun member => untouched (List.mem_cons_of_mem _ member))
private theorem heap_refl (heap : Mutable.Heap) : HeapFrame heap heap := fun _ _ found => found
private theorem heap_trans (first : HeapFrame before middle) (second : HeapFrame middle after) : HeapFrame before after :=
  fun address object found => second address object (first address object found)

private structure Locals (target : Int) (lower arrayAddress powerAddress primeAddress : Nat) (env : HeapCalls.Env) : Prop where
  getTarget : env "target" = some (number target)
  getLower : env "lower" = some (number lower)
  getArray : env "primes" = some (.ref arrayAddress)
  getPowers : env "pow10" = some (.ref powerAddress)
  getPrimes : env "is_prime" = some (.ref primeAddress)

private theorem Locals.preserved (locals : Locals target lower arrayAddress powerAddress primeAddress before)
    (saved : Stable before after) : Locals target lower arrayAddress powerAddress primeAddress after :=
  ⟨(saved "target" (by decide)).trans locals.getTarget, (saved "lower" (by decide)).trans locals.getLower,
    (saved "primes" (by decide)).trans locals.getArray, (saved "pow10" (by decide)).trans locals.getPowers,
    (saved "is_prime" (by decide)).trans locals.getPrimes⟩

private structure Snapshot (items : List Nat) (width arrayAddress powerAddress primeAddress : Nat)
    (bytes : List UInt8) (heap : Mutable.Heap) : Prop where
  array : heap[arrayAddress]? = some (.list (numbers items))
  powers : heap[powerAddress]? = some (.list (numbers (powers width)))
  primes : heap[primeAddress]? = some (.bytearray bytes)
  size : bytes.length = 10 ^ width
  correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)

private theorem Snapshot.preserved (snapshot : Snapshot items width arrayAddress powerAddress primeAddress bytes before)
    (saved : HeapFrame before after) : Snapshot items width arrayAddress powerAddress primeAddress bytes after :=
  ⟨saved _ _ snapshot.array, saved _ _ snapshot.powers, saved _ _ snapshot.primes, snapshot.size, snapshot.correct⟩

private def Scan (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap) : Option Nat → Prop
  | none => ∃ finalEnv finalHeap, BlockContinues program env body heap finalEnv finalHeap ∧
      Stable env finalEnv ∧ HeapFrame heap finalHeap
  | some answer => ∃ finalHeap, (∀ following, BlockReturns program env (body ++ following) heap (number answer) finalHeap) ∧
      HeapFrame heap finalHeap

private theorem scan_prepend (continued : BlockContinues program env body heap middleEnv middleHeap)
    (stable : Stable env middleEnv) (saved : HeapFrame heap middleHeap)
    (executed : Scan middleEnv following middleHeap answer) : Scan env (body ++ following) heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finalHeap, finished, preserved, framed⟩ := executed
      exact ⟨finalEnv, finalHeap, continued.append finished, stable_trans stable preserved, heap_trans saved framed⟩
  | some answer =>
      obtain ⟨finalHeap, returned, framed⟩ := executed
      refine ⟨finalHeap, ?_, heap_trans saved framed⟩
      intro after
      simpa only [List.append_assoc] using continued.returns (returned after)

private theorem scan_branch {flag : Bool} (evaluated : Evaluates program env condition heap value heap)
    (truth : Mutable.truth value heap = some (.ok flag, heap))
    (executed : Scan env ((if flag then yes else no) ++ rest) heap answer) :
    Scan env (.ifThen condition yes no :: rest) heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finalHeap, finished, preserved, framed⟩ := executed
      exact ⟨finalEnv, finalHeap, BlockContinues.branch_next evaluated truth finished, preserved, framed⟩
  | some answer =>
      obtain ⟨finalHeap, returned, framed⟩ := executed
      refine ⟨finalHeap, ?_, framed⟩
      intro after
      exact BlockReturns.branch evaluated truth (by simpa only [List.append_assoc] using returned after)

private theorem scan_continue (target : Calls.loopTarget body = some (next, rest))
    (executed : Scan env (next :: rest) heap answer) : Scan env (.continueLoop :: body) heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finalHeap, finished, preserved, framed⟩ := executed
      exact ⟨finalEnv, finalHeap, BlockContinues.continue_loop target finished, preserved, framed⟩
  | some answer =>
      obtain ⟨finalHeap, returned, framed⟩ := executed
      refine ⟨finalHeap, ?_, framed⟩
      intro after
      exact BlockReturns.continue_loop (Calls.loopTarget_append target after)
        (by simpa only [List.cons_append] using returned after)

private theorem scan_resume (executed : Scan env [next] heap answer) : Scan env [.loopResume next] heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finalHeap, finished, preserved, framed⟩ := executed
      exact ⟨finalEnv, finalHeap, BlockContinues.loop_resume finished, preserved, framed⟩
  | some answer =>
      obtain ⟨finalHeap, returned, framed⟩ := executed
      exact ⟨finalHeap, fun after => BlockReturns.loop_resume (returned after), framed⟩

private theorem scan_sequence_step (source : SequenceSource) (position : Nat) (value : Mutable.Value)
    (sequence : Mutable.sequence (sequenceValue source) heap = some (.ok items, heap))
    (item : items[position]? = some value)
    (executed : Scan (setLocal env "p" value)
      (body ++ [Calls.loopContinuation body (.sequenceStepLoop "p" source (position+1) body [])]) heap answer) :
    Scan env [.sequenceStepLoop "p" source position body []] heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finalHeap, finished, preserved, framed⟩ := executed
      exact ⟨finalEnv, finalHeap, BlockContinues.sequence_step sequence item finished,
        stable_trans (stable_set env value) preserved, framed⟩
  | some answer =>
      obtain ⟨finalHeap, returned, framed⟩ := executed
      refine ⟨finalHeap, ?_, framed⟩
      intro after
      apply BlockReturns.sequence_step sequence item
      simpa only [List.append_assoc] using returned after

private def accepted (target : Int) (lower p : Nat) : Bool :=
  decide (lower ≤ p) && ProjectEulerStatements.P51.hasPrimeFamily target.toNat p

private theorem lower_test (p lower : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getP : env "p" = some (number p)) (getLower : env "lower" = some (number lower)) :
    Evaluates program env (.compare (.name "p") [(.lt, .name "lower")]) heap (flag (decide (p < lower))) heap := by
  apply Evaluates.compare (av := number p) (bv := number lower) (middle := heap) (afterRight := heap)
    (Evaluates.lookup getP) (Evaluates.lookup getLower)
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

/-- A decimal interval determines the candidate's width and the shared sieve/table dimensions. -/
theorem digits_width (p width : Nat) (positive : 1 ≤ width) (lower : 10 ^ (width - 1) ≤ p) (upper : p < 10 ^ width) :
    (Nat.digits 10 p).length = width := by
  have atLeast := (Nat.lt_digits_length_iff (b := 10) (k := width-1) (by decide) p).mpr lower
  have atMost := (Nat.digits_length_le_iff (b := 10) (k := width) (by decide) p).mpr upper
  omega

private theorem loop (items remaining : List Nat) (position width lower arrayAddress powerAddress primeAddress : Nat)
    (target : Int) (primes : ∀ p ∈ items, Nat.Prime p)
    (widths : ∀ p ∈ items, lower ≤ p → (Nat.digits 10 p).length = width)
    (suffix : items.drop position = remaining) (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (locals : Locals target lower arrayAddress powerAddress primeAddress env)
    (snapshot : Snapshot items width arrayAddress powerAddress primeAddress bytes heap) :
    Scan env [.sequenceStepLoop "p" (.ref arrayAddress) position primeBody []] heap
      (remaining.find? (accepted target lower)) := by
  have sequence : Mutable.sequence (.ref arrayAddress) heap = some (.ok (numbers items), heap) := by
    simp [Mutable.sequence, snapshot.array]
  induction remaining generalizing position env heap with
  | nil =>
      refine ⟨env, heap, ?_, stable_refl env, heap_refl heap⟩
      apply BlockContinues.sequence_done (source := .ref arrayAddress) sequence
      · have missing : items[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [numbers, missing]
      · exact .nil
  | cons head tail ih =>
      have item : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      have member := List.mem_of_getElem? item
      have suffixTail : items.drop (position+1) = tail := by simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := setLocal env "p" (number head)
      let next := Stmt.sequenceStepLoop "p" (.ref arrayAddress) (position+1) primeBody []
      have boundLocals := locals.preserved (stable_set env (number head))
      have tested := lower_test head lower bound heap rfl boundLocals.getLower
      by_cases skipped : head < lower
      · have rejected : accepted target lower head = false := by
          simp only [accepted, show ¬ lower ≤ head by omega, decide_false, Bool.false_and]
        rw [List.find?_cons, rejected]
        apply scan_sequence_step (.ref arrayAddress) position (number head) sequence (by simp [numbers, item])
        change Scan bound (primeBody ++ [.loopResume next]) heap (tail.find? (accepted target lower))
        apply scan_branch (flag := true) (value := flag true)
          (by simpa only [skipped, decide_true] using tested) (by simp [Mutable.truth, scalarTruth])
        exact scan_continue (next := next) (rest := []) rfl
          (ih (position+1) suffixTail bound heap boundLocals snapshot sequence)
      · have prime := primes head member
        have width := widths head member (by omega)
        obtain ⟨middleHeap, returned, continued, framed⟩ := candidate.result head powerAddress primeAddress target prime bound heap bytes
          rfl boundLocals.getTarget boundLocals.getPowers boundLocals.getPrimes
          (by simpa only [width] using snapshot.powers) snapshot.primes
          (by simpa only [width] using snapshot.size) snapshot.correct
        cases family : ProjectEulerStatements.P51.hasPrimeFamily target.toNat head with
        | true =>
            have matched : accepted target lower head = true := by
              simp only [accepted, show lower ≤ head by omega, decide_true, Bool.true_and, family]
            rw [List.find?_cons, matched]
            apply scan_sequence_step (.ref arrayAddress) position (number head) sequence (by simp [numbers, item])
            change Scan bound (primeBody ++ [.loopResume next]) heap (some head)
            apply scan_branch (flag := false) (value := flag false)
              (by simpa only [skipped, decide_false] using tested) (by simp [Mutable.truth, scalarTruth])
            refine ⟨middleHeap, ?_, framed⟩
            intro following
            simpa only [List.append_assoc] using returned family ([.loopResume next] ++ following)
        | false =>
            have rejected : accepted target lower head = false := by
              simp only [accepted, family, Bool.and_false]
            obtain ⟨middleEnv, executed, preserved⟩ := continued family
            have stable := stable_candidate preserved
            have nextLocals := boundLocals.preserved stable
            have nextSnapshot := snapshot.preserved framed
            have nextSequence : Mutable.sequence (.ref arrayAddress) middleHeap = some (.ok (numbers items), middleHeap) := by
              simp [Mutable.sequence, nextSnapshot.array]
            have rest := ih (position+1) suffixTail middleEnv middleHeap nextLocals nextSnapshot nextSequence
            rw [List.find?_cons, rejected]
            apply scan_sequence_step (.ref arrayAddress) position (number head) sequence (by simp [numbers, item])
            change Scan bound (primeBody ++ [.loopResume next]) heap (tail.find? (accepted target lower))
            apply scan_branch (flag := false) (value := flag false)
              (by simpa only [skipped, decide_false] using tested) (by simp [Mutable.truth, scalarTruth])
            change Scan bound (primeBody.drop 1 ++ [.loopResume next]) heap _
            exact scan_prepend executed stable framed (scan_resume rest)

private theorem scan_for_sequence (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (evaluated : Evaluates program env iterable heap value heap)
    (sequence : Mutable.sequence value heap = some (.ok items, heap))
    (executed : Scan env [.sequenceStepLoop "p" (sequenceSource value) 0 body []] heap answer) :
    Scan env [.forIn (.name "p") iterable body []] heap answer := by
  cases answer with
  | none =>
      obtain ⟨finalEnv, finalHeap, finished, preserved, framed⟩ := executed
      exact ⟨finalEnv, finalHeap, BlockContinues.for_sequence ordinary evaluated sequence finished, preserved, framed⟩
  | some answer =>
      obtain ⟨finalHeap, returned, framed⟩ := executed
      exact ⟨finalHeap, fun after => BlockReturns.for_sequence ordinary evaluated sequence (returned after), framed⟩

/-- The source prime loop returns the first qualifying in-range prime and preserves all previously allocated objects. -/
theorem result (items : List Nat) (width lower arrayAddress powerAddress primeAddress : Nat) (target : Int)
    (primes : ∀ p ∈ items, Nat.Prime p)
    (widths : ∀ p ∈ items, lower ≤ p → (Nat.digits 10 p).length = width)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (bytes : List UInt8)
    (getTarget : env "target" = some (number target)) (getLower : env "lower" = some (number lower))
    (getArray : env "primes" = some (.ref arrayAddress)) (getPowers : env "pow10" = some (.ref powerAddress))
    (getPrimes : env "is_prime" = some (.ref primeAddress))
    (arrayFound : heap[arrayAddress]? = some (.list (numbers items)))
    (powersFound : heap[powerAddress]? = some (.list (numbers (powers width))))
    (primesFound : heap[primeAddress]? = some (.bytearray bytes)) (size : bytes.length = 10 ^ width)
    (correct : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Scan env [digitBody[4]'(by decide)] heap (items.find? (accepted target lower)) := by
  have scanned := loop items items 0 width lower arrayAddress powerAddress primeAddress target primes widths rfl env heap bytes
    ⟨getTarget, getLower, getArray, getPowers, getPrimes⟩ ⟨arrayFound, powersFound, primesFound, size, correct⟩
  exact scan_for_sequence (value := .ref arrayAddress) (items := numbers items)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getArray)
    (by simp [Mutable.sequence, arrayFound]) scanned

end primeSearch

namespace digitSearch
open ProjectEulerSolutions
private def upper (width : Nat) : Nat := 10 ^ width - 1
private def lower (width : Nat) : Nat := 10 ^ (width - 1)
private def powerExpression : Expr := .binOp (.constant (.int 10)) .pow
  (.binOp (.binOp (.name "ndigits") .sub (.constant (.int 1))) .sub (.name "i"))
private def powerBody (address : Nat) : List Stmt := [.comprehensionAppend address powerExpression]

private theorem ten_power (exponent : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) (expression : Expr)
    (evaluated : Evaluates program env expression heap (number exponent) heap) :
    Evaluates program env (.binOp (.constant (.int 10)) .pow expression) heap (number (10 ^ exponent : Nat)) heap := by
  exact Evaluates.binary (av := number 10) (bv := number exponent) (middle := heap) (afterRight := heap)
    Evaluates.constant evaluated (by simp [Mutable.binary, scalarBinary, show ¬ (exponent : Int) < 0 by omega])

private theorem power_value (width index : Nat) (inside : index < width) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getWidth : env "ndigits" = some (number width)) (getIndex : env "i" = some (number index)) :
    Evaluates program env powerExpression heap (number (10 ^ (width - 1 - index) : Nat)) heap := by
  have exponent : Evaluates program env (.binOp (.binOp (.name "ndigits") .sub (.constant (.int 1))) .sub (.name "i"))
      heap (number ((width : Int) - 1 - index)) heap := by
    apply Evaluates.binary (av := number ((width : Int) - 1)) (bv := number index) (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (av := number width) (bv := number 1) (middle := heap) (afterRight := heap)
        (Evaluates.lookup getWidth) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
    · exact Evaluates.lookup getIndex
    · simp [Mutable.binary, scalarBinary]
  have nonnegative : (width : Int) - 1 - index = ((width - 1 - index : Nat) : Int) := by omega
  rw [nonnegative] at exponent
  exact ten_power (width - 1 - index) env heap _ exponent

private theorem power_loop (width first remaining : Nat) (endpoint : first + remaining = width)
    (base : Mutable.Heap) (before : List Nat) (env : HeapCalls.Env) (getWidth : env "ndigits" = some (number width)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "i" first width 1 (powerBody base.length) []]
      (HeapCollection.working base (numbers before) []) finalEnv
      (HeapCollection.working base (numbers (before ++ (List.range' first remaining).map (fun index => 10 ^ (width - 1 - index)))) []) := by
  induction remaining generalizing first before env with
  | zero =>
      refine ⟨env, ?_⟩
      simp only [List.range'_zero, List.map_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ remaining ih =>
      let bound := setLocal env "i" (number first)
      have boundWidth : bound "ndigits" = some (number width) := by simpa [bound, setLocal] using getWidth
      have evaluated := power_value width first (by omega) bound (HeapCollection.working base (numbers before) []) boundWidth rfl
      have emitted := HeapCollection.emit base (numbers before) [] [] (number (10 ^ (width - 1 - first) : Nat))
        (by simpa only [List.append_nil] using evaluated)
      have stored : BlockContinues program bound (powerBody base.length) (HeapCollection.working base (numbers before) []) bound
          (HeapCollection.working base (numbers (before ++ [10 ^ (width - 1 - first)])) []) := by
        simpa only [powerBody, numbers, List.map_append, List.map_singleton, List.nil_append] using emitted
      obtain ⟨finalEnv, executed⟩ := ih (first+1) (by omega) (before ++ [10 ^ (width - 1 - first)]) bound boundWidth
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [powerBody, List.range'_succ, List.map_cons, List.append_assoc, List.singleton_append] using stored.append executed

/-- The source comprehension builds the complete big-endian decimal-weight table. -/
theorem power_table (width : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) (getWidth : env "ndigits" = some (number width)) :
    Evaluates program env (.listComp powerExpression [("i", .call (.name "range") [.name "ndigits"], [])])
      heap (.ref heap.length) (heap ++ [.list (numbers (powers width))]) := by
  obtain ⟨finalEnv, executed⟩ := power_loop width 0 width (by omega) heap []
    (comprehensionEnv env [("i", .call (.name "range") [.name "ndigits"], [])]) (by simpa [comprehensionEnv] using getWidth)
  apply Evaluates.list_comp_range (.cons (Evaluates.lookup getWidth) .nil) rfl
  simpa only [List.nil_append, ← List.range_eq_range', powers, HeapCollection.working] using executed

private def boundsEnv (env : HeapCalls.Env) (width : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "upper" (number (upper width))) "lower" (number (lower width))
private def sievedEnv (env : HeapCalls.Env) (base width : Nat) : HeapCalls.Env :=
  setLocal (setLocal (boundsEnv env width) "is_prime" (.ref (base+1))) "primes" (.ref (base+2))
private def preparedEnv (env : HeapCalls.Env) (base width : Nat) : HeapCalls.Env :=
  setLocal (sievedEnv env base width) "pow10" (.ref (base+4))
private def sievedHeap (heap : Mutable.Heap) (bytes : List UInt8) (width : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray bytes, .list (numbers (sieve.primes (upper width))),
    .tuple [.ref (heap.length+1), .ref (heap.length+2)]]
private def preparedHeap (heap : Mutable.Heap) (bytes : List UInt8) (width : Nat) : Mutable.Heap :=
  sievedHeap heap bytes width ++ [.list (numbers (powers width))]

/-- The actual width setup computes its bounds, unpacks the inclusive source sieve, and allocates its power table. -/
theorem prepare (width : Nat) (positive : 1 ≤ width) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getWidth : env "ndigits" = some (number width)) :
    ∃ bytes, BlockContinues program env (digitBody.take 4) heap (preparedEnv env heap.length width) (preparedHeap heap bytes width) ∧
      bytes.length = 10 ^ width ∧ (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  have powerBound : 10 ≤ 10 ^ width := by
    simpa only [pow_one] using Nat.pow_le_pow_right (by decide : 1 ≤ 10) positive
  have large : 1 ≤ upper width := by unfold upper; omega
  let withUpper := setLocal env "upper" (number (upper width))
  have first : BlockContinues program env [digitBody[0]'(by decide)] heap withUpper heap := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number (10 ^ width : Nat)) (bv := number 1) (middle := heap) (afterRight := heap)
      (ten_power width env heap (.name "ndigits") (Evaluates.lookup getWidth)) Evaluates.constant
    simp [upper, Mutable.binary, scalarBinary, Int.natCast_sub (show 1 ≤ 10 ^ width by omega)]
  have exponent : Evaluates program withUpper (.binOp (.name "ndigits") .sub (.constant (.int 1))) heap (number (width-1 : Nat)) heap :=
    Evaluates.binary (av := number width) (bv := number 1) (middle := heap) (afterRight := heap)
      (Evaluates.lookup (by simpa [withUpper, setLocal] using getWidth)) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, Int.natCast_sub positive])
  have second : BlockContinues program withUpper [digitBody[1]'(by decide)] heap (boundsEnv env width) heap :=
    BlockContinues.assign (ten_power (width-1) withUpper heap _ exponent)
  obtain ⟨bytes, returned, length, correct⟩ := sieve.positive_result (upper width) large heap
  have called : ReturnsValue program "sieve" [number (upper width)] heap (.ref (heap.length+3)) (sievedHeap heap bytes width) := by
    simpa only [sievedHeap, sieve.numbers, numbers] using returned
  have third : BlockContinues program (boundsEnv env width) [digitBody[2]'(by decide)] heap
      (sievedEnv env heap.length width) (sievedHeap heap bytes width) := by
    apply BlockContinues.unpack (names := ["is_prime", "primes"]) (items := [.ref (heap.length+1), .ref (heap.length+2)])
      (value := .ref (heap.length+3)) (middle := sievedHeap heap bytes width) (finalHeap := sievedHeap heap bytes width) rfl
    · exact Evaluates.call (.cons (Evaluates.lookup rfl) .nil) called
    · simp [Mutable.sequence, sievedHeap]
    · rfl
  have fourth : BlockContinues program (sievedEnv env heap.length width) [digitBody[3]'(by decide)] (sievedHeap heap bytes width)
      (preparedEnv env heap.length width) (preparedHeap heap bytes width) := by
    have evaluated := power_table width (sievedEnv env heap.length width) (sievedHeap heap bytes width)
      (by simpa [sievedEnv, boundsEnv, setLocal] using getWidth)
    simpa [preparedEnv, preparedHeap, sievedHeap] using BlockContinues.assign (name := "pow10") evaluated
  refine ⟨bytes, first.append (second.append (third.append fourth)), ?_, correct⟩
  dsimp only [upper] at length
  omega

private def writes : List String := ["upper", "lower", "is_prime", "primes", "pow10"] ++ primeSearch.writes
private def Stable (before after : HeapCalls.Env) : Prop := ∀ name, name ∉ writes → after name = before name
private def answer (target : Int) (width : Nat) : Option Nat :=
  (sieve.primes (upper width)).find? (primeSearch.accepted target (lower width))
private def Scan (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap) : Option Nat → Prop
  | none => ∃ finalEnv finalHeap, BlockContinues program env body heap finalEnv finalHeap ∧
      Stable env finalEnv ∧ primeSearch.HeapFrame heap finalHeap
  | some value => ∃ finalHeap, (∀ following, BlockReturns program env (body ++ following) heap (number value) finalHeap) ∧
      primeSearch.HeapFrame heap finalHeap

/-- A complete source width iteration prepares its tables and scans the entire prime window. -/
theorem result (width : Nat) (positive : 1 ≤ width) (target : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getWidth : env "ndigits" = some (number width)) (getTarget : env "target" = some (number target)) :
    Scan env digitBody heap (answer target width) := by
  obtain ⟨bytes, prepared, size, correct⟩ := prepare width positive env heap getWidth
  have allPrimes : ∀ p ∈ sieve.primes (upper width), Nat.Prime p := fun p member => (sieve.mem_primes _ p).mp member |>.1
  have allWidths : ∀ p ∈ sieve.primes (upper width), lower width ≤ p → (Nat.digits 10 p).length = width := by
    intro p member minimum
    have maximum := (sieve.mem_primes _ p).mp member |>.2
    apply primeSearch.digits_width p width positive minimum
    have powerPositive : 0 < 10 ^ width := Nat.pow_pos (by decide)
    unfold upper at maximum
    omega
  have oldHeap : primeSearch.HeapFrame heap (preparedHeap heap bytes width) := by
    intro address object found
    exact HeapCollection.prefix_found (sievedHeap heap bytes width) _ address object
      (HeapCollection.prefix_found heap _ address object found)
  have scanned := primeSearch.result (sieve.primes (upper width)) width (lower width)
    (heap.length+2) (heap.length+4) (heap.length+1) target allPrimes allWidths
    (preparedEnv env heap.length width) (preparedHeap heap bytes width) bytes
    (by simpa [preparedEnv, sievedEnv, boundsEnv, setLocal] using getTarget)
    (by simp [preparedEnv, sievedEnv, boundsEnv, setLocal])
    (by simp [preparedEnv, sievedEnv, setLocal]) (by simp [preparedEnv, setLocal])
    (by simp [preparedEnv, sievedEnv, setLocal])
    (by simp [preparedHeap, sievedHeap]) (by simp [preparedHeap, sievedHeap]) (by simp [preparedHeap, sievedHeap]) size correct
  change primeSearch.Scan (preparedEnv env heap.length width) [digitBody[4]'(by decide)] (preparedHeap heap bytes width) (answer target width) at scanned
  cases found : answer target width with
  | some value =>
      rw [found] at scanned
      obtain ⟨finalHeap, returned, framed⟩ := scanned
      refine ⟨finalHeap, ?_, primeSearch.heap_trans oldHeap framed⟩
      intro following
      exact prepared.returns (returned following)
  | none =>
      rw [found] at scanned
      obtain ⟨finalEnv, finalHeap, continued, preserved, framed⟩ := scanned
      refine ⟨finalEnv, finalHeap, prepared.append continued, ?_, primeSearch.heap_trans oldHeap framed⟩
      intro name untouched
      have notPrimeWrite : name ∉ primeSearch.writes := fun member => untouched (List.mem_append_right _ member)
      have noUpper : name ≠ "upper" := fun equal => untouched (by simp [writes, equal])
      have noLower : name ≠ "lower" := fun equal => untouched (by simp [writes, equal])
      have noBuffer : name ≠ "is_prime" := fun equal => untouched (by simp [writes, equal])
      have noArray : name ≠ "primes" := fun equal => untouched (by simp [writes, equal])
      have noPower : name ≠ "pow10" := fun equal => untouched (by simp [writes, equal])
      exact (preserved name notPrimeWrite).trans
        (by simp only [preparedEnv, sievedEnv, boundsEnv, setLocal, noUpper, noLower, noBuffer, noArray, noPower, if_false])

end digitSearch

namespace minimum
private abbrev reference := data.«project-euler-statements».data.py.project_euler_statements.p51.naive.answer
private abbrev width (target : Int) := (Nat.digits 10 (reference target)).length

private theorem prime (target : Int) (supported : target ≤ 8) : Nat.Prime (reference target) := by
  have family := data.«project-euler-statements».data.py.project_euler_statements.p51.naive.answer_family target supported
  simp only [ProjectEulerStatements.P51.hasPrimeFamily, Bool.and_eq_true, ProjectEulerStatements.P51.isPrime, decide_eq_true_eq] at family
  exact family.2

private theorem find_least (predicate : Nat → Bool) (winner : Nat) : ∀ items : List Nat,
    items.Pairwise (· < ·) → winner ∈ items → predicate winner = true →
      (∀ p ∈ items, predicate p = true → winner ≤ p) → items.find? predicate = some winner := by
  intro items
  induction items with
  | nil => intro _ member; simp at member
  | cons head tail ih =>
      intro ordered member good least
      obtain ⟨before, rest⟩ := List.pairwise_cons.mp ordered
      cases accepted : predicate head with
      | true =>
          have lower := least head (by simp) accepted
          have upper : head ≤ winner := by
            rcases List.mem_cons.mp member with same | present
            · omega
            · exact Nat.le_of_lt (before winner present)
          have same : head = winner := by omega
          simp only [List.find?_cons, accepted]
          exact congrArg some same
      | false =>
          have present : winner ∈ tail := by
            rcases List.mem_cons.mp member with same | present
            · rw [same, accepted] at good
              contradiction
            · exact present
          have found := ih rest present good (fun p member => least p (by simp [member]))
          simpa only [List.find?_cons, accepted, Bool.false_eq_true, if_false] using found

/-- A sorted prime-window scan finds the reference minimum as soon as its upper bound reaches it. -/
theorem window (target : Int) (supported : target ≤ 8) (digits : Nat) (lower : digitSearch.lower digits ≤ reference target) :
    digitSearch.answer target digits = if reference target ≤ digitSearch.upper digits then some (reference target) else none := by
  by_cases inside : reference target ≤ digitSearch.upper digits
  · rw [if_pos inside]
    apply find_least (primeSearch.accepted target (digitSearch.lower digits)) (reference target)
      (sieve.primes (digitSearch.upper digits)) (sieve.primes_pairwise _)
    · exact (sieve.mem_primes _ _).mpr ⟨prime target supported, inside⟩
    · simp only [primeSearch.accepted, lower, decide_true, Bool.true_and]
      exact data.«project-euler-statements».data.py.project_euler_statements.p51.naive.answer_family target supported
    · intro p _ accepted
      simp only [primeSearch.accepted, Bool.and_eq_true, decide_eq_true_eq] at accepted
      exact data.«project-euler-statements».data.py.project_euler_statements.p51.naive.answer_minimal target supported p accepted.2
  · rw [if_neg inside]
    apply List.find?_eq_none.mpr
    intro p member accepted
    have upper := (sieve.mem_primes _ p).mp member |>.2
    simp only [primeSearch.accepted, Bool.and_eq_true, decide_eq_true_eq] at accepted
    have least : reference target ≤ p := data.«project-euler-statements».data.py.project_euler_statements.p51.naive.answer_minimal target supported p accepted.2
    omega

/-- The checked 121313 witness confines the original source search to its first six widths. -/
theorem width_bounds (target : Int) (supported : target ≤ 8) : 1 ≤ width target ∧ width target ≤ 6 := by
  have positive := (prime target supported).pos
  have nonempty := (Nat.digits_ne_nil_iff_ne_zero (b := 10)).mpr (by omega : reference target ≠ 0)
  refine ⟨by simpa only [width] using List.length_pos_iff.mpr nonempty, ?_⟩
  apply (Nat.digits_length_le_iff (b := 10) (k := 6) (by decide) (reference target)).mpr
  have maximum : reference target ≤ 121313 := data.«project-euler-statements».data.py.project_euler_statements.p51.naive.answer_le_witness target supported
  norm_num
  omega

private theorem lower_bound (target : Int) (supported : target ≤ 8) (first : Nat) (before : first ≤ width target) :
    digitSearch.lower first ≤ reference target := by
  have widths := width_bounds target supported
  have leading : 10 ^ (width target - 1) ≤ reference target :=
    (Nat.lt_digits_length_iff (b := 10) (k := width target-1) (by decide) (reference target)).mp (by change width target - 1 < width target; omega)
  exact (Nat.pow_le_pow_right (by decide : 1 ≤ 10) (show first-1 ≤ width target-1 by omega)).trans leading

private theorem inside_width (target : Int) (supported : target ≤ 8) : reference target ≤ digitSearch.upper (width target) := by
  have upper : reference target < 10 ^ width target :=
    (Nat.digits_length_le_iff (b := 10) (k := width target) (by decide) (reference target)).mp (Nat.le_refl _)
  have positive := (prime target supported).pos
  unfold digitSearch.upper
  omega

/-- The source outer loop reaches the least qualifying prime within six decimal widths. -/
private theorem widths_loop (target : Int) (supported : target ≤ 8) (first : Nat)
    (positive : 1 ≤ first) (before : first ≤ width target) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getTarget : env "target" = some (number target)) :
    ∃ finalHeap, ∀ following, BlockReturns program env
      (.rangeStepLoop "ndigits" first 10 1 digitBody [] :: following) heap (number (reference target)) finalHeap := by
  have bounds := width_bounds target supported
  have active : Calls.rangeContinues first 10 1 = true := by
    simp [Calls.rangeContinues]
    omega
  let boundEnv := setLocal env "ndigits" (number first)
  have scanned := digitSearch.result first positive target boundEnv heap
    (by simp [boundEnv, setLocal]) (by simpa [boundEnv, setLocal] using getTarget)
  have selected := window target supported first (lower_bound target supported first before)
  by_cases inside : reference target ≤ digitSearch.upper first
  · rw [if_pos inside] at selected
    rw [selected] at scanned
    obtain ⟨finalHeap, returned, _⟩ := scanned
    refine ⟨finalHeap, ?_⟩
    intro following
    apply BlockReturns.range_step active
    exact returned (.rangeStepLoop "ndigits" ((first : Int)+1) 10 1 digitBody [] :: following)
  · rw [if_neg inside] at selected
    rw [selected] at scanned
    obtain ⟨finalEnv, middle, continued, stable, _⟩ := scanned
    have earlier : first < width target := by
      have reached := inside_width target supported
      by_contra late
      have same : first = width target := by omega
      exact inside (same ▸ reached)
    have preserved : finalEnv "target" = some (number target) :=
      (stable "target" (by decide)).trans (by simpa [boundEnv, setLocal] using getTarget)
    obtain ⟨finalHeap, returned⟩ := widths_loop target supported (first+1) (by omega) (by omega) finalEnv middle preserved
    refine ⟨finalHeap, ?_⟩
    intro following
    apply BlockReturns.range_step active
    have rest := returned following
    have joined := continued.returns rest
    simpa only [Nat.cast_add, Nat.cast_one] using joined
termination_by width target - first

end minimum

/-- The executable Python solver terminates and returns the reference minimum for every supported target. -/
theorem result (target : Int) (supported : target ≤ 8) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number target] heap (number (minimum.reference target)) finalHeap := by
  let entry := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number target] []
  obtain ⟨finalHeap, returned⟩ := minimum.widths_loop target supported 1 (by decide)
    (minimum.width_bounds target supported).1 entry heap rfl
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  exact BlockReturns.for_range (.cons Evaluates.constant (.cons Evaluates.constant .nil)) rfl
    (returned (code.drop 1))

private abbrev inputEnv (target : Int) : ScalarEnv :=
  fun name => if name = "target" then some (.int target) else none

private theorem target_bound (target : Int)
    (condition : Calls.Condition program (inputEnv target) Solvers.P51Ast.requires_solve_0) : target ≤ 8 := by
  have bound := Calls.Evaluates.condition_int_compare condition (by rfl : inputEnv target "target" = some (.int target))
  simpa only [evalCmpOp, decide_eq_true_eq] using bound

theorem «Equiv(naive)» (target : Int)
    (condition : Calls.Condition program (inputEnv target) Solvers.P51Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int target]).Equiv
      (HeapCalls.scalarFunction P51Ast.module "naive" [.int target]) := by
  have bound := target_bound target condition
  obtain ⟨finalHeap, returned⟩ := result target bound []
  obtain ⟨extra, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p51.naive.result target bound []
  have optimized := returned.scalar_finishes (values := [.int target])
  have original := reference.scalar_finishes (values := [.int target])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem «Supported» :
    (∀ target : Int, Calls.Condition program (inputEnv target) Solvers.P51Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int target]).Supported) ∧
    (∀ target : Int, Calls.Condition program (inputEnv target) Solvers.P51Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P51Ast.module "naive" [.int target]).Supported) := by
  constructor
  · intro target condition
    obtain ⟨finalHeap, returned⟩ := result target (target_bound target condition) []
    exact (returned.scalar_returns (values := [.int target])).supported
  · intro target condition
    obtain ⟨extra, returned⟩ := data.«project-euler-statements».data.py.project_euler_statements.p51.naive.result
      target (target_bound target condition) []
    exact (returned.scalar_returns (values := [.int target])).supported

end solve

end solvers.«51»
