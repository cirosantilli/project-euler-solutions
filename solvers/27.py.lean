import «.formalsw».solvers.«27.py.ast»
import lang.py.FormalSW.DictionaryCalls
import ProjectEulerSolutions.Lib.PrimeSieve
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import ProjectEulerStatements.Lib.QuadraticPrimes
import ProjectEulerStatements.Lib.QuadraticPrimeWinner
import data.«project-euler-statements».data.py.project_euler_statements.«p27.py»

-- Keep the large independent proof sections within a single elaboration task.
set_option Elab.async false

/-! The full bytearray sieve used by problem 27, including its shared-heap execution. -/
namespace solvers.«27».Sieve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P27Ast.module
private abbrev num (n : Int) : Mutable.Value := .scalar (.int n)

open ProjectEulerSolutions.PrimeSieve

private def sieveBody : List Stmt :=
  match findFunctionIn program.body "sieve_primes" with
  | some (_, body) => executableBody body
  | none => []
private def loopBody : List Stmt :=
  match (sieveBody[4]? : Option Stmt) with
  | some (.whileDo _ body _) => body
  | _ => []
private def markBody : List Stmt :=
  match (loopBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []
private def markEnv (env : HeapCalls.Env) (p : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "step" (num p)) "start" (num (p*p : Nat))

theorem mark_result (env : HeapCalls.Env) (heap : Mutable.Heap) (address limit p : Nat)
    (bytes : List UInt8) (lower : 2 ≤ p) (inside : p*p ≤ limit) (size : bytes.length = limit+1)
    (getP : env "p" = some (num p)) (getLimit : env "limit" = some (num limit))
    (getBuffer : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes)) :
    BlockContinues program env markBody heap (markEnv env p) (heap.set address (.bytearray (mark bytes p))) := by
  let e1 := setLocal env "step" (num p)
  let e2 := markEnv env p
  let count : Int := ((limit : Int)-p*p)/p+1
  have first : BlockContinues program env [markBody[0]'(by decide)] heap e1 heap :=
    BlockContinues.assign (Evaluates.lookup getP)
  have second : BlockContinues program e1 [markBody[1]'(by decide)] heap e2 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup (by simpa [e1, setLocal] using getP))
      (Evaluates.lookup (by simpa [e1, setLocal] using getP)) (by simp [Mutable.binary, scalarBinary, num])
  have third : BlockContinues program e2 [markBody[2]'(by decide)] heap e2
      (heap.set address (.bytearray (mark bytes p))) := by
    apply BlockContinues.setSlice (value := .scalar (.bytes (List.replicate count.toNat 0)))
      (afterRhs := heap) (afterTarget := heap) (receiver := .ref address)
      (lowerValue := num (p*p : Nat)) (upperValue := num (limit+1 : Nat)) (strideValue := num p)
    · apply Evaluates.binary (av := .scalar (.bytes [0])) (bv := num count) (middle := heap) (afterRight := heap)
      · exact Evaluates.constant
      · apply Evaluates.binary (av := num (((limit : Int)-p*p)/p)) (bv := num 1)
          (middle := heap) (afterRight := heap)
        · apply Evaluates.binary (av := num ((limit : Int)-p*p)) (bv := num p)
            (middle := heap) (afterRight := heap)
          · exact Evaluates.binary (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getLimit))
              (Evaluates.lookup rfl) (by simp [Mutable.binary, scalarBinary, num])
          · exact Evaluates.lookup (by simp [e2, markEnv, setLocal])
          · simp [Mutable.binary, scalarBinary, num, show p ≠ 0 by omega, show 0 < p by omega]
        · exact Evaluates.constant
        · rfl
      · simp [Mutable.binary, scalarBinary]
    · exact .cons (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getBuffer))
        (.cons (Evaluates.lookup rfl)
          (.cons (Evaluates.binary (Evaluates.lookup (by simpa [e2, markEnv, setLocal] using getLimit))
            Evaluates.constant (by simp [Mutable.binary, scalarBinary, num]))
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
  exact first.append (second.append third)

private theorem heap_inside {heap : Mutable.Heap} {address : Nat}
    (found : heap[address]? = some (obj : Mutable.Object)) : address < heap.length := by
  by_contra invalid
  rw [List.getElem?_eq_none (by omega)] at found
  cases found
private theorem heap_set_same (heap : Mutable.Heap) (address : Nat) (obj : Mutable.Object)
    (found : heap[address]? = some obj) : heap.set address obj = heap := by
  apply List.ext_getElem?
  intro i
  by_cases same : i = address
  · subst i
    rw [List.getElem?_set_self (heap_inside found), found]
  · simp [List.getElem?_set_ne (Ne.symm same)]

private theorem read_buffer (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat)
    (bytes : List UInt8) (valid : p < bytes.length)
    (getP : env "p" = some (num p)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    Evaluates program env (.subscript (.name "is_prime") (.name "p")) heap (num bytes[p].toNat) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro h; cases h) (Evaluates.lookup getBuffer) (Evaluates.lookup getP)
  exact Mutable.getItem_buffer heap address p bytes found valid

private def Stable (before after : HeapCalls.Env) : Prop :=
  after "limit" = before "limit" ∧ after "is_prime" = before "is_prime"
private theorem iteration (limit p : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (lower : 2 ≤ p) (inside : p*p ≤ limit)
    (getP : env "p" = some (num p)) (getLimit : env "limit" = some (num limit))
    (getBuffer : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes))
    (sieved : Sieved limit p bytes) :
    ∃ finalEnv finalBytes, BlockContinues program env loopBody heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Sieved limit (p+1) finalBytes ∧
      finalEnv "p" = some (num (p+1 : Nat)) ∧ Stable env finalEnv := by
  have valid : p < bytes.length := by rw [sieved.1]; nlinarith
  have live : bytes[p].toNat ≠ 0 ↔ Nat.Prime p :=
    (sieved.2 p bytes[p] lower (List.getElem?_eq_getElem valid)).trans (survives_current p lower)
  by_cases prime : Nat.Prime p
  · let nextEnv := setLocal (markEnv env p) "p" (num (p+1 : Nat))
    refine ⟨nextEnv, mark bytes p, ?_, sieved_mark limit p bytes sieved prime, rfl, ?_⟩
    · have marked := mark_result env heap address limit p bytes lower inside sieved.1 getP getLimit getBuffer found
      have branch : BlockContinues program env [loopBody[0]'(by decide)] heap (markEnv env p)
          (heap.set address (.bytearray (mark bytes p))) := by
        apply BlockContinues.branch (flag := true) (middle := heap) (read_buffer heap env address p bytes valid getP getBuffer found)
        · simp [Mutable.truth, scalarTruth, live.mpr prime]
        · exact marked
      apply branch.append
      apply BlockContinues.augAssign (value := num (p+1 : Nat))
        (finalHeap := heap.set address (.bytearray (mark bytes p)))
        (by simpa [markEnv, setLocal] using getP) Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, num]
    · simp [Stable, nextEnv, markEnv, setLocal]
  · let nextEnv := setLocal env "p" (num (p+1 : Nat))
    refine ⟨nextEnv, bytes, ?_, sieved_skip limit p bytes sieved prime, rfl, ?_⟩
    · rw [heap_set_same heap address _ found]
      have branch : BlockContinues program env [loopBody[0]'(by decide)] heap env heap := by
        apply BlockContinues.branch (flag := false) (middle := heap) (read_buffer heap env address p bytes valid getP getBuffer found)
        · have zero : bytes[p].toNat = 0 := by tauto
          simp [Mutable.truth, scalarTruth, zero]
        · exact .nil
      apply branch.append
      apply BlockContinues.augAssign (value := num (p+1 : Nat)) (finalHeap := heap) getP Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary, num]
    · simp [Stable, nextEnv, setLocal]

private theorem loop (limit p : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (lower : 2 ≤ p)
    (getP : env "p" = some (num p)) (getLimit : env "limit" = some (num limit))
    (getBuffer : env "is_prime" = some (.ref address)) (found : heap[address]? = some (.bytearray bytes))
    (sieved : Sieved limit p bytes) :
    ∃ finalEnv finalBytes finalP, BlockContinues program env [sieveBody[4]'(by decide)] heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Sieved limit finalP finalBytes ∧
      limit < finalP^2 ∧ Stable env finalEnv := by
  induction measure : limit+1-p using Nat.strong_induction_on generalizing p bytes env heap with
  | h size ih =>
      by_cases active : p*p ≤ limit
      · obtain ⟨middleEnv, middleBytes, iterated, nextSieved, nextP, stable⟩ :=
          iteration limit p bytes env heap address lower active getP getLimit getBuffer found sieved
        let middleHeap := heap.set address (.bytearray middleBytes)
        have middleFound : middleHeap[address]? = some (.bytearray middleBytes) :=
          List.getElem?_set_self (heap_inside found)
        have progress : limit+1-(p+1) < size := by
          have : p ≤ limit := by nlinarith
          omega
        obtain ⟨finalEnv, finalBytes, finalP, executed, finalSieved, stopped, saved⟩ :=
          ih _ progress (p+1) middleBytes middleEnv middleHeap (by omega) nextP
            (stable.1.trans getLimit) (stable.2.trans getBuffer) middleFound nextSieved rfl
        refine ⟨finalEnv, finalBytes, finalP, ?_, finalSieved, stopped,
          saved.1.trans stable.1, saved.2.trans stable.2⟩
        apply BlockContinues.while_loop (flag := true) (value := .scalar (.bool true)) (middle := heap)
        · apply Evaluates.compare
            (Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP) rfl)
            (Evaluates.lookup getLimit)
          simp [HeapCalls.compare, scalarCompare, evalCmpOp]
          exact_mod_cast active
        · rfl
        · simpa [loopBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            middleHeap, List.set_set] using iterated.append executed
      · refine ⟨env, bytes, p, ?_, sieved, by nlinarith, rfl, rfl⟩
        rw [heap_set_same heap address _ found]
        apply BlockContinues.while_loop (flag := false) (value := .scalar (.bool false)) (middle := heap)
        · apply Evaluates.compare
            (Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP) rfl)
            (Evaluates.lookup getLimit)
          simp [HeapCalls.compare, scalarCompare, evalCmpOp]
          exact_mod_cast Nat.lt_of_not_ge active
        · rfl
        · exact .nil

private theorem initial_slice (limit : Nat) (large : 2 ≤ limit) :
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

private def initializedEnv (env : HeapCalls.Env) (address : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "is_prime" (.ref address)) "p" (num 2)
private def initialHeap (heap : Mutable.Heap) (limit : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray (initial limit)]
private theorem setup (limit : Nat) (large : 2 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (num limit)) :
    BlockContinues program env ((sieveBody.drop 1).take 3) heap
      (initializedEnv env (heap.length+1)) (initialHeap heap limit) := by
  let e1 := setLocal env "is_prime" (.ref (heap.length+1))
  let allocated := heap ++ [.bytearray [1], .bytearray (List.replicate (limit+1) 1)]
  have first : BlockContinues program env [sieveBody[1]'(by decide)] heap e1 allocated := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := .ref heap.length) (middle := heap ++ [.bytearray [1]])
      (bv := num (limit+1 : Nat)) (afterRight := heap ++ [.bytearray [1]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant (by simp [Mutable.binary, scalarBinary, num])
    · simp [allocated, Mutable.binary, Mutable.allocateBuffer, Except.map,
        show (BinOp.mult == .mult) = true from rfl]
  have second : BlockContinues program e1 [sieveBody[2]'(by decide)] allocated e1 (initialHeap heap limit) := by
    have found : allocated[heap.length+1]? = some (.bytearray (List.replicate (limit+1) 1)) := by simp [allocated]
    have write := Mutable.setSlice_buffer allocated (heap.length+1) (List.replicate (limit+1) 1) [0,0] (initial limit)
      (.scalar (.int 0)) (.scalar (.int 2)) (.scalar .none)
      { start := some 0, stop := some 2 } found rfl (initial_slice limit large)
    have same : allocated.set (heap.length+1) (.bytearray (initial limit)) = initialHeap heap limit := by
      simp [allocated, initialHeap]
    rw [same] at write
    exact BlockContinues.setSlice Evaluates.constant
      (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))) write
  have third : BlockContinues program e1 [sieveBody[3]'(by decide)] (initialHeap heap limit)
      (initializedEnv env (heap.length+1)) (initialHeap heap limit) :=
    BlockContinues.assign Evaluates.constant
  exact first.append (second.append third)

/-- The source builds a fresh bytearray whose live entries are exactly the primes. -/
theorem prepare (limit : Nat) (large : 2 ≤ limit) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (num limit)) :
    ∃ finalEnv bytes,
      BlockContinues program env ((sieveBody.drop 1).take 4) heap finalEnv
        (heap ++ [.bytearray [1], .bytearray bytes]) ∧
      finalEnv "limit" = some (num limit) ∧ finalEnv "is_prime" = some (.ref (heap.length+1)) ∧
      bytes.length = limit+1 ∧
      (∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  let initialized := initializedEnv env (heap.length+1)
  have getLimit' : initialized "limit" = some (num limit) := by simpa [initialized, initializedEnv, setLocal] using getLimit
  have found : (initialHeap heap limit)[heap.length+1]? = some (.bytearray (initial limit)) := by simp [initialHeap]
  obtain ⟨finalEnv, bytes, p, executed, sieved, finished, stable⟩ :=
    loop limit 2 (initial limit) initialized (initialHeap heap limit) (heap.length+1) (by omega)
      rfl getLimit' rfl found (sieved_initial limit large)
  refine ⟨finalEnv, bytes, ?_, stable.1.trans getLimit', stable.2, sieved.1, ?_⟩
  · have run := (setup limit large env heap getLimit).append executed
    simpa [initialHeap, List.set_append] using run
  · intro i byte lower found
    have bound : i < bytes.length := (List.getElem?_eq_some_iff.mp found).1
    exact (sieved.2 i byte lower found).trans (survives_finished limit p i lower (by rw [sieved.1] at bound; omega) finished)

/-- Increasing primes in the requested inclusive interval. -/
def primes (limit : Nat) : List Nat := (List.range' 2 (limit-1)).filter Nat.Prime
def numbers (values : List Nat) : List Mutable.Value := values.map (fun (n : Nat) => num n)
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
      let bound := setLocal env "i" (num start)
      have getBuffer' : bound "is_prime" = some (.ref address) := by simpa [bound, setLocal] using getBuffer
      have read : Evaluates program bound (.subscript (.name "is_prime") (.name "i")) heap
          (num bytes[start].toNat) heap := by
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
              (Evaluates.lookup (globals := []) (module := program) (heap := heap) (show bound "i" = some (num start) from rfl)) collected
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
  .listComp (.name "i") [("i", .call (.name "range") [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))],
    [.subscript (.name "is_prime") (.name "i")])]
private theorem collect (limit : Nat) (large : 2 ≤ limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (bytes : List UInt8)
    (getLimit : env "limit" = some (num limit)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (size : bytes.length = limit+1)
    (correct : ∀ i byte, 2 ≤ i → bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    Evaluates program env outputExpression heap (.ref heap.length) (heap ++ [.list (numbers (primes limit))]) := by
  have bound := heap_inside found
  obtain ⟨finalEnv, collected⟩ := collect_loop (limit-1) 2 limit (by omega) (by omega)
    (heap ++ [.list []])
    (comprehensionEnv env [("i", .call (.name "range") [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))],
      [.subscript (.name "is_prime") (.name "i")])]) address heap.length (by omega) bytes []
    (by simpa [comprehensionEnv] using getBuffer)
    (by rw [List.getElem?_append_left bound]; exact found) size correct (by simp [numbers])
  apply Evaluates.list_comp_range (values := [num 2, num (limit+1 : Nat)]) (middle := heap)
  · exact .cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup getLimit)
      Evaluates.constant (by simp [Mutable.binary, scalarBinary, num])) .nil)
  · rfl
  · simpa [primes, selected, List.set_append, comprehensionFilters, comprehensionBody, collectBody] using collected

/-- The actual Python sieve returns every prime up to the bound, in ascending order. -/
theorem result (limit : Nat) (large : 2 ≤ limit) (heap : Mutable.Heap) :
    ∃ bytes, ReturnsValue program "sieve_primes" [num limit] heap (.ref (heap.length+2))
      (heap ++ [.bytearray [1], .bytearray bytes, .list (numbers (primes limit))]) := by
  let env := argumentEnv [⟨"limit", some (.name "int"), none⟩] [num limit]
  obtain ⟨finalEnv, bytes, prepared, getLimit, getBuffer, size, correct⟩ := prepare limit large env heap rfl
  let middle := heap ++ [Mutable.Object.bytearray [1], .bytearray bytes]
  have found : middle[heap.length+1]? = some (.bytearray bytes) := by simp [middle]
  have collected := collect limit large middle finalEnv (heap.length+1) bytes getLimit getBuffer found size correct
  refine ⟨bytes, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap)
  · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  · rfl
  · have returned := prepared.returns (BlockReturns.return_value (rest := []) collected)
    simpa [middle] using returned

theorem mem_primes (limit p : Nat) (large : 2 ≤ limit) : p ∈ primes limit ↔ Nat.Prime p ∧ p ≤ limit := by
  simp only [primes, List.mem_filter, List.mem_range', decide_eq_true_eq]
  constructor
  · rintro ⟨⟨i, bound, rfl⟩, prime⟩; exact ⟨prime, by omega⟩
  · rintro ⟨prime, upper⟩; exact ⟨⟨p-2, by have := prime.two_le; omega, by have := prime.two_le; omega⟩, prime⟩

theorem primes_pairwise (limit : Nat) : (primes limit).Pairwise (· < ·) :=
  (List.pairwise_lt_range' (s := 2) (n := limit-1)).filter _

end solvers.«27».Sieve

namespace solvers.«27»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P27Ast.module

namespace PrimeCache

/-- Negative integers and zero have a false primality value as well. -/
def answer (n : Int) : Bool := decide (Nat.Prime n.toNat)

private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev truth (n : Int) : Mutable.Value := .scalar (.bool (answer n))

/-- Every stored entry records the mathematical primality of its integer key. -/
def Valid (entries : List (Constant × Mutable.Value)) : Prop :=
  ∀ key value, (key, value) ∈ entries → ∃ n : Int, key = .int n ∧ value = truth n

theorem valid_empty : Valid [] := by
  intro key value member
  simp at member

theorem Valid.lookup (valid : Valid entries) (n : Int)
    (found : ScalarDict.lookup entries (.int n) = some value) : value = truth n := by
  induction entries with
  | nil => simp [ScalarDict.lookup] at found
  | cons pair rest ih =>
      obtain ⟨key, previous⟩ := pair
      obtain ⟨m, rfl, rfl⟩ := valid key previous (by simp)
      have tail : Valid rest := fun key value member => valid key value (by simp [member])
      by_cases same : m = n
      · subst m
        simpa [ScalarDict.lookup, scalarCompare, evalCmpOp] using found.symm
      · simp [ScalarDict.lookup, scalarCompare, evalCmpOp, same] at found
        exact ih tail found

theorem Valid.insert (valid : Valid entries) (n : Int) :
    Valid (ScalarDict.insert entries (.int n) (truth n)) := by
  induction entries with
  | nil =>
      intro key value member
      simp only [ScalarDict.insert, List.mem_singleton, Prod.mk.injEq] at member
      obtain ⟨rfl, rfl⟩ := member
      exact ⟨n, rfl, rfl⟩
  | cons pair rest ih =>
      obtain ⟨key, previous⟩ := pair
      obtain ⟨m, rfl, rfl⟩ := valid key previous (by simp)
      have tail : Valid rest := fun key value member => valid key value (by simp [member])
      intro key value member
      by_cases same : m = n
      · subst m
        simp [ScalarDict.insert, scalarCompare, evalCmpOp] at member
        rcases member with ⟨rfl, rfl⟩ | member
        · exact ⟨n, rfl, rfl⟩
        · exact tail key value member
      · simp [ScalarDict.insert, scalarCompare, evalCmpOp, same] at member
        rcases member with ⟨rfl, rfl⟩ | member
        · exact ⟨m, rfl, rfl⟩
        · exact ih tail key value member

theorem Valid.classify (valid : Valid entries) (n : Int) :
    ScalarDict.lookup entries (.int n) = none ∨ ScalarDict.lookup entries (.int n) = some (truth n) := by
  cases found : ScalarDict.lookup entries (.int n) with
  | none => exact Or.inl rfl
  | some value => exact Or.inr (congrArg some (valid.lookup n found))

/-- Checked prime divisors below the current trial value have all been excluded. -/
def AvoidBelow (n bound : Nat) : Prop := ∀ q : Nat, Nat.Prime q → q < bound → ¬ q ∣ n

theorem prime_of_square (n p : Nat) (large : 2 ≤ n) (square : n < p * p)
    (tested : AvoidBelow n p) : Nat.Prime n := by
  by_contra composite
  have prime := Nat.minFac_prime (by omega : n ≠ 1)
  have small := Nat.minFac_sq_le_self (by omega : 0 < n) composite
  have below : n.minFac < p := by nlinarith
  exact tested n.minFac prime below (Nat.minFac_dvd n)

theorem composite_of_divisor (n p : Nat) (large : 2 ≤ p) (square : p * p ≤ n)
    (divides : p ∣ n) : ¬ Nat.Prime n :=
  Nat.not_prime_of_dvd_of_lt divides large (by nlinarith)

theorem AvoidBelow.odd_step (tested : AvoidBelow n p) (large : 3 ≤ p)
    (odd : p % 2 = 1) (missing : ¬ p ∣ n) : AvoidBelow n (p + 2) := by
  intro q prime before
  by_cases below : q < p
  · exact tested q prime below
  · have oddq : q % 2 = 1 := by
      rcases prime.eq_two_or_odd with two | oddq
      · omega
      · exact oddq
    have same : q = p := by omega
    simpa [same] using missing

/-- A sorted, complete initial segment of the primes, ending at an odd prime. -/
structure TrialPrimes (values : List Nat) (last : Nat) : Prop where
  ordered : values.Pairwise (· < ·)
  primes : ∀ p ∈ values, Nat.Prime p
  bounded : ∀ p ∈ values, p ≤ last
  lastValue : values.getLast? = some last
  large : 3 ≤ last
  odd : last % 2 = 1
  complete : ∀ q : Nat, Nat.Prime q → q ≤ last → q ∈ values

theorem TrialPrimes.covered_next (trials : TrialPrimes values last)
    (q : Nat) (prime : Nat.Prime q) (below : q < last + 2) : q ∈ values := by
  apply trials.complete q prime
  by_contra larger
  have oddq : q % 2 = 1 := by
    rcases prime.eq_two_or_odd with two | oddq
    · have := trials.large; omega
    · exact oddq
  have := trials.odd
  omega

theorem TrialPrimes.prefix_avoid (trials : TrialPrimes values last)
    (split : values = processed ++ p :: rest)
    (checked : ∀ q ∈ processed, ¬ q ∣ n) : AvoidBelow n p := by
  intro q prime before
  have current : p ∈ values := by simp [split]
  have bounded := trials.bounded p current
  have present := trials.covered_next q prime (by omega)
  rw [split, List.mem_append] at present
  rcases present with earlier | later
  · exact checked q earlier
  · have ordered : (p :: rest).Pairwise (· < ·) :=
      (List.pairwise_append.mp (split ▸ trials.ordered)).2.1
    rcases List.mem_cons.mp later with same | member
    · omega
    · have := List.rel_of_pairwise_cons ordered member
      omega

theorem TrialPrimes.finished_avoid (trials : TrialPrimes values last)
    (checked : ∀ q ∈ values, ¬ q ∣ n) : AvoidBelow n (last + 2) :=
  fun q prime below => checked q (trials.covered_next q prime below)

theorem answer_small (n : Int) (small : n < 2) : answer n = false := by
  have tooSmall : n.toNat < 2 := by omega
  have composite : ¬ Nat.Prime n.toNat := fun prime => by have := prime.two_le; omega
  simp [answer, composite]

def store (heap : Mutable.Heap) (address : Nat) (entries : List (Constant × Mutable.Value))
    (n : Int) : Mutable.Heap :=
  heap.set address (.dict (ScalarDict.insert entries (.int n) (truth n)))

theorem store_lookup (heap : Mutable.Heap) (address : Nat) (entries : List (Constant × Mutable.Value))
    (found : heap[address]? = some (.dict entries)) (n : Int) :
    (store heap address entries n)[address]? =
      some (.dict (ScalarDict.insert entries (.int n) (truth n))) := by
  have bounds : address < heap.length := List.getElem?_eq_some_iff.mp found |>.1
  simp [store, bounds]

theorem store_other (heap : Mutable.Heap) (address other : Nat) (entries : List (Constant × Mutable.Value))
    (different : address ≠ other) (n : Int) :
    (store heap address entries n)[other]? = heap[other]? := by
  simp [store, List.getElem?_set_ne different]


/-- A successful dictionary probe returns the recorded value without touching either input object. -/
theorem hit (n : Int) (heap : Mutable.Heap) (trialsAddress cacheAddress : Nat)
    (entries : List (Constant × Mutable.Value))
    (found : heap[cacheAddress]? = some (.dict entries))
    (known : ScalarDict.lookup entries (.int n) = some (truth n)) :
    ReturnsValue program "is_prime" [number n, .ref trialsAddress, .ref cacheAddress] heap (truth n) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (middle := heap) (flag := true) (value := .scalar (.bool true))
  · apply Evaluates.membership (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil))
    simp [HeapCalls.membership, found, known, number]
  · rfl
  · apply BlockReturns.return_value
    apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup rfl) (Evaluates.lookup rfl)
    exact Mutable.getItem_dictionary heap cacheAddress entries found (.int n) known

private theorem store_constant (n : Int) (flag : Bool) (env : Mutable.Env) (heap : Mutable.Heap)
    (address : Nat) (entries : List (Constant × Mutable.Value)) (following : List Stmt)
    (getN : env "n" = some (number n)) (getCache : env "cache" = some (.ref address))
    (found : heap[address]? = some (.dict entries)) :
    BlockReturns program env
      (.assign [.subscript (.name "cache") (.name "n")] (.constant (.bool flag)) ::
        .return (.constant (.bool flag)) :: following)
      heap (.scalar (.bool flag)) (heap.set address (.dict (ScalarDict.insert entries (.int n) (.scalar (.bool flag))))) := by
  apply BlockReturns.setItem (by intro a b c; simp) Evaluates.constant
    (.cons (Evaluates.lookup getCache) (.cons (Evaluates.lookup getN) .nil))
    (Mutable.setItem_dictionary heap address entries found (.int n) (.scalar (.bool flag)))
  exact BlockReturns.return_value Evaluates.constant

/-- The small-integer miss branch writes False, including for negative keys. -/
theorem small (n : Int) (small : n < 2) (heap : Mutable.Heap) (trialsAddress cacheAddress : Nat)
    (entries : List (Constant × Mutable.Value))
    (found : heap[cacheAddress]? = some (.dict entries))
    (missing : ScalarDict.lookup entries (.int n) = none) :
    ReturnsValue program "is_prime" [number n, .ref trialsAddress, .ref cacheAddress] heap (truth n)
      (store heap cacheAddress entries n) := by
  simp only [truth, answer_small n small, store]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (middle := heap) (flag := false) (value := .scalar (.bool false))
  · apply Evaluates.membership (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil))
    simp [HeapCalls.membership, found, missing, number]
  · rfl
  · apply BlockReturns.branch (middle := heap) (flag := true) (value := .scalar (.bool true))
    · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small])
    · rfl
    · exact store_constant n false _ heap cacheAddress entries _ rfl rfl found

private def fallbackLoop : List Stmt :=
  (executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2).drop 4

/-- The fallback tests all remaining odd candidates and preserves the shared cache until returning. -/
theorem fallback (n p : Nat) (largeN : 2 ≤ n) (largeP : 3 ≤ p)
    (odd : p % 2 = 1) (tested : AvoidBelow n p)
    (env : Mutable.Env) (heap : Mutable.Heap) (address : Nat) (entries : List (Constant × Mutable.Value))
    (getN : env "n" = some (number n)) (getP : env "p" = some (number p))
    (getCache : env "cache" = some (.ref address)) (found : heap[address]? = some (.dict entries)) :
    BlockReturns program env fallbackLoop heap (truth n) (store heap address entries n) := by
  induction measure : n + 1 - p using Nat.strong_induction_on generalizing p env with
  | h measure ih =>
      have condition : Evaluates program env
          (.compare (.binOp (.name "p") .mult (.name "p")) [(.ltE, .name "n")])
          heap (.scalar (.bool (decide (p * p ≤ n)))) heap := by
        apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number ((p * p : Nat) : Int))
          (Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP)
            (by simp [Mutable.binary, scalarBinary, number])) (Evaluates.lookup getN)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]; norm_cast
      by_cases square : p * p ≤ n
      · apply BlockReturns.while_loop (flag := true) (by simpa [square] using condition) rfl
        have divides : Evaluates program env
            (.compare (.binOp (.name "n") .mod (.name "p")) [(.eq, .constant (.int 0))])
            heap (.scalar (.bool (decide (p ∣ n)))) heap := by
          apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number ((n % p : Nat) : Int))
            (Evaluates.binary (Evaluates.lookup getN) (Evaluates.lookup getP)
              (by simp [Mutable.binary, scalarBinary, number, show p ≠ 0 by omega, show 0 < p by omega]))
            Evaluates.constant
          simp [HeapCalls.compare, scalarCompare, evalCmpOp, ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast]
        by_cases divisor : p ∣ n
        · have composite := composite_of_divisor n p (by omega) square divisor
          simp only [truth, answer, Int.toNat_natCast, composite, decide_false, store]
          apply BlockReturns.branch (middle := heap) (flag := true) (by simpa [divisor] using divides) rfl
          exact store_constant n false env heap address entries _ getN getCache found
        · apply BlockReturns.branch (middle := heap) (flag := false) (by simpa [divisor] using divides) rfl
          apply BlockReturns.augAssign getP Evaluates.constant (value := number (p + 2)) (afterOperation := heap)
          · simp [inplace, Mutable.binary, scalarBinary, number]
          · have bounded : p ≤ n := by nlinarith
            apply ih (n + 1 - (p + 2)) (by omega) (p + 2) (by omega) (by omega)
              (tested.odd_step largeP odd divisor)
            all_goals simp [setLocal, getN, getCache, number]
      · have prime := prime_of_square n p largeN (by omega) tested
        simp only [truth, answer, Int.toNat_natCast, prime, decide_true, store]
        apply BlockReturns.while_loop (flag := false) (by simpa [square] using condition) rfl
        exact store_constant n true env heap address entries _ getN getCache found


abbrev numbers (values : List Nat) : List Mutable.Value := values.map fun (n : Nat) => number n

private theorem last_item (values : List Nat) (last : Nat) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (numbers values))) (lastValue : values.getLast? = some last) :
    Mutable.getItem (.ref address) (number (-1)) heap = some (.ok (number last), heap) := by
  have nonempty : values ≠ [] := by intro empty; simp [empty] at lastValue
  have positive : 0 < values.length := List.length_pos_iff.mpr nonempty
  have nonnegative : ¬ (-1 : Int) + values.length < 0 := by omega
  have inside : ¬ (-1 : Int) + values.length ≥ values.length := by omega
  have normalized : ((-1 : Int) + values.length).toNat = values.length - 1 := by omega
  simp [Mutable.getItem, Mutable.index, found, number, numbers, nonempty, normalized,
    ← List.getLast?_eq_getElem?, lastValue, Except.map]

private def trialBody : List Stmt :=
  let statements : List Stmt := executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2
  match statements[2]? with
  | some (Stmt.forIn _ _ body _) => body
  | _ => []

private def afterTrials : List Stmt :=
  (executableBody ((findFunctionIn program.body "is_prime").getD ([], [])).2).drop 3

private theorem trials_loop (n last : Nat) (largeN : 2 ≤ n) (values processed remaining : List Nat)
    (trials : TrialPrimes values last) (split : values = processed ++ remaining)
    (checked : ∀ q ∈ processed, ¬ q ∣ n) (env : Mutable.Env) (heap : Mutable.Heap)
    (trialsAddress cacheAddress : Nat) (entries : List (Constant × Mutable.Value))
    (getN : env "n" = some (number n)) (getTrials : env "trial_primes" = some (.ref trialsAddress))
    (getCache : env "cache" = some (.ref cacheAddress))
    (trialsFound : heap[trialsAddress]? = some (.list (numbers values)))
    (cacheFound : heap[cacheAddress]? = some (.dict entries)) :
    BlockReturns program env
      (.sequenceStepLoop "p" (.ref trialsAddress) processed.length trialBody [] :: afterTrials)
      heap (truth n) (store heap cacheAddress entries n) := by
  induction remaining generalizing processed env with
  | nil =>
      have same : values = processed := by simpa using split
      apply BlockReturns.sequence_done (items := numbers values)
        (by simp [sequenceValue, Mutable.sequence, trialsFound])
        (by simp [numbers, same])
      apply BlockReturns.assign (middle := heap) (value := number (last + 2))
      · apply Evaluates.binary (av := number last)
          (Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup getTrials)
            (Evaluates.unary Evaluates.constant rfl) (last_item values last heap trialsAddress trialsFound trials.lastValue))
          Evaluates.constant
        simp [Mutable.binary, scalarBinary, number]
      · apply fallback n (last + 2) largeN (by have := trials.large; omega)
          (by have := trials.odd; omega) (trials.finished_avoid (by simpa [same] using checked))
        all_goals simp [setLocal, getN, getCache, number, cacheFound]
  | cons p remaining ih =>
      have present : p ∈ values := by simp [split]
      have primeP := trials.primes p present
      have largeP := primeP.two_le
      have below := trials.prefix_avoid split checked
      apply BlockReturns.sequence_step (items := numbers values)
        (by simp [sequenceValue, Mutable.sequence, trialsFound])
        (value := number p) (by simp [numbers, split])
      have squareTest : Evaluates program (setLocal env "p" (number p))
          (.compare (.binOp (.name "p") .mult (.name "p")) [(.gt, .name "n")])
          heap (.scalar (.bool (decide (n < p * p)))) heap := by
        apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number ((p * p : Nat) : Int))
          (Evaluates.binary (Evaluates.lookup (value := number p) (by simp [setLocal])) (Evaluates.lookup (value := number p) (by simp [setLocal]))
            (by simp [Mutable.binary, scalarBinary, number]))
          (Evaluates.lookup (value := number n) (by simpa [setLocal] using getN))
        simp [HeapCalls.compare, scalarCompare, evalCmpOp]; norm_cast
      by_cases square : n < p * p
      · have primeN := prime_of_square n p largeN square below
        simp only [truth, answer, Int.toNat_natCast, primeN, decide_true, store]
        apply BlockReturns.branch (middle := heap) (flag := true) (by simpa [square] using squareTest) rfl
        exact store_constant n true _ heap cacheAddress entries _
          (by simpa [setLocal] using getN) (by simpa [setLocal] using getCache) cacheFound
      · apply BlockReturns.branch (middle := heap) (flag := false) (by simpa [square] using squareTest) rfl
        have divides : Evaluates program (setLocal env "p" (number p))
            (.compare (.binOp (.name "n") .mod (.name "p")) [(.eq, .constant (.int 0))])
            heap (.scalar (.bool (decide (p ∣ n)))) heap := by
          apply Evaluates.compare (middle := heap) (afterRight := heap) (av := number ((n % p : Nat) : Int))
            (Evaluates.binary (Evaluates.lookup (value := number n) (by simpa [setLocal] using getN))
              (Evaluates.lookup (value := number p) (by simp [setLocal]))
              (by simp [Mutable.binary, scalarBinary, number, show p ≠ 0 by omega, show 0 < p by omega]))
            Evaluates.constant
          simp [HeapCalls.compare, scalarCompare, evalCmpOp, ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast]
        by_cases divisor : p ∣ n
        · have composite := composite_of_divisor n p largeP (by omega) divisor
          have different : n ≠ p := by nlinarith
          simp only [truth, answer, Int.toNat_natCast, composite, decide_false, store]
          apply BlockReturns.branch (middle := heap) (flag := true) (by simpa [divisor] using divides) rfl
          apply BlockReturns.setItem (afterRhs := heap) (afterTarget := heap) (by intro a b c; simp)
            (value := .scalar (.bool false))
            (afterWrite := heap.set cacheAddress (.dict (ScalarDict.insert entries (.int n) (.scalar (.bool false)))))
          · exact Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup (value := number n) (by simpa [setLocal] using getN))
              (Evaluates.lookup (value := number p) (by simp [setLocal]))
              (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, different])
          · exact .cons (Evaluates.lookup (value := .ref cacheAddress) (by simpa [setLocal] using getCache))
              (.cons (Evaluates.lookup (value := number n) (by simpa [setLocal] using getN)) .nil)
          · exact Mutable.setItem_dictionary heap cacheAddress entries cacheFound (.int n) (.scalar (.bool false))
          · apply BlockReturns.return_value
            apply Evaluates.subscript (by intro a b c; simp) (Evaluates.lookup (value := .ref cacheAddress) (by simpa [setLocal] using getCache))
              (Evaluates.lookup (value := number n) (by simpa [setLocal] using getN))
            apply Mutable.getItem_dictionary _ cacheAddress
              (ScalarDict.insert entries (.int n) (.scalar (.bool false))) _ (.int n)
              (ScalarDict.lookup_insert_self entries (.int n) (.scalar (.bool false)))
            have bound := (List.getElem?_eq_some_iff.mp cacheFound).1
            simp [bound]
        · apply BlockReturns.branch (middle := heap) (flag := false) (by simpa [divisor] using divides) rfl
          have later : ∀ q ∈ processed ++ [p], ¬ q ∣ n := by
            intro q member
            rcases List.mem_append.mp member with earlier | current
            · exact checked q earlier
            · simpa using (List.mem_singleton.mp current) ▸ divisor
          have run := ih (processed ++ [p]) (by simpa [List.append_assoc] using split) later
            (setLocal env "p" (number p)) (by simpa [setLocal] using getN)
            (by simpa [setLocal] using getTrials) (by simpa [setLocal] using getCache)
          simpa only [List.length_append, List.length_singleton] using run

/-- A complete prime prefix and the odd fallback together decide primality for any integer. -/
theorem result (n : Int) (last : Nat) (values : List Nat) (trials : TrialPrimes values last)
    (heap : Mutable.Heap) (trialsAddress cacheAddress : Nat) (entries : List (Constant × Mutable.Value))
    (valid : Valid entries)
    (trialsFound : heap[trialsAddress]? = some (.list (numbers values)))
    (cacheFound : heap[cacheAddress]? = some (.dict entries)) :
    ∃ finalHeap, ReturnsValue program "is_prime" [number n, .ref trialsAddress, .ref cacheAddress]
      heap (truth n) finalHeap ∧
      (finalHeap = heap ∨ finalHeap = store heap cacheAddress entries n) := by
  rcases valid.classify n with missing | known
  · refine ⟨store heap cacheAddress entries n, ?_, Or.inr rfl⟩
    by_cases smallN : n < 2
    · exact small n smallN heap trialsAddress cacheAddress entries cacheFound missing
    · lift n to Nat using (by omega : 0 ≤ n)
      apply BlockReturns.function rfl rfl
      apply BlockReturns.branch (middle := heap) (flag := false) (value := .scalar (.bool false))
      · apply Evaluates.membership (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil))
        simp [HeapCalls.membership, cacheFound, missing, number]
      · rfl
      · apply BlockReturns.branch (middle := heap) (flag := false) (value := .scalar (.bool false))
        · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
        · rfl
        · apply BlockReturns.for_sequence (items := numbers values) (middle := heap) (by intro expressions; simp)
            (Evaluates.lookup rfl) (by simp [Mutable.sequence, trialsFound])
          exact trials_loop n last (by omega) values [] values trials rfl (by simp)
            _ heap trialsAddress cacheAddress entries rfl rfl rfl trialsFound cacheFound
  · exact ⟨heap, hit n heap trialsAddress cacheAddress entries cacheFound known, Or.inl rfl⟩


/-- Repeated calls retain the cache invariant and every unrelated heap object. -/
theorem result_preserves (n : Int) (last : Nat) (values : List Nat) (trials : TrialPrimes values last)
    (heap : Mutable.Heap) (trialsAddress cacheAddress : Nat) (entries : List (Constant × Mutable.Value))
    (valid : Valid entries)
    (trialsFound : heap[trialsAddress]? = some (.list (numbers values)))
    (cacheFound : heap[cacheAddress]? = some (.dict entries)) :
    ∃ finalHeap finalEntries,
      ReturnsValue program "is_prime" [number n, .ref trialsAddress, .ref cacheAddress]
        heap (truth n) finalHeap ∧
      Valid finalEntries ∧
      finalHeap[cacheAddress]? = some (.dict finalEntries) ∧
      finalHeap[trialsAddress]? = some (.list (numbers values)) ∧
      (∀ other, cacheAddress ≠ other → finalHeap[other]? = heap[other]?) ∧
      finalHeap.length = heap.length := by
  obtain ⟨finalHeap, run, unchanged | inserted⟩ :=
    result n last values trials heap trialsAddress cacheAddress entries valid trialsFound cacheFound
  · subst finalHeap
    exact ⟨heap, entries, run, valid, cacheFound, trialsFound, fun _ _ => rfl, rfl⟩
  · subst finalHeap
    have different : cacheAddress ≠ trialsAddress := by
      intro same
      rw [same, trialsFound] at cacheFound
      simp at cacheFound
    refine ⟨store heap cacheAddress entries n, ScalarDict.insert entries (.int n) (truth n),
      run, valid.insert n, store_lookup heap cacheAddress entries cacheFound n, ?_, ?_, ?_⟩
    · rw [store_other heap cacheAddress trialsAddress entries different n]
      exact trialsFound
    · intro other distinct
      exact store_other heap cacheAddress other entries distinct n
    · simp [store]

theorem answer_eq_isPrimeInt (n : Int) : answer n = ProjectEulerStatements.P27.isPrimeInt n := by
  by_cases prime : Nat.Prime n.toNat
  · have positive : 1 < n := by have := prime.two_le; omega
    simp [answer, ProjectEulerStatements.P27.isPrimeInt, prime, positive]
  · simp [answer, ProjectEulerStatements.P27.isPrimeInt, prime]

end PrimeCache
namespace Consecutive
open ProjectEulerStatements.P27 ProjectEulerStatements.QuadraticPrimes
private abbrev num (n : Int) : Mutable.Value := .scalar (.int n)


private def countLoop : List Stmt :=
  (executableBody ((findFunctionIn program.body "consecutive_prime_length").getD ([], [])).2).drop 1

private theorem count_loop (a b : Int) (n : Nat) (before : n ≤ first a b)
    (last : Nat) (values : List Nat) (trials : PrimeCache.TrialPrimes values last)
    (env : Mutable.Env) (getA : env "a" = some (num a)) (getB : env "b" = some (num b))
    (getN : env "n" = some (num n)) (trialsAddress cacheAddress : Nat)
    (getTrials : env "trial_primes" = some (.ref trialsAddress))
    (getCache : env "cache" = some (.ref cacheAddress))
    (heap : Mutable.Heap) (entries : List (Constant × Mutable.Value))
    (valid : PrimeCache.Valid entries)
    (trialsFound : heap[trialsAddress]? = some (.list (PrimeCache.numbers values)))
    (cacheFound : heap[cacheAddress]? = some (.dict entries)) :
    ∃ finalHeap finalEntries,
      BlockReturns program env countLoop heap (num (first a b)) finalHeap ∧
      PrimeCache.Valid finalEntries ∧
      finalHeap[cacheAddress]? = some (.dict finalEntries) ∧
      finalHeap[trialsAddress]? = some (.list (PrimeCache.numbers values)) ∧
      (∀ other, cacheAddress ≠ other → finalHeap[other]? = heap[other]?) ∧
      finalHeap.length = heap.length := by
  induction measure : first a b - n using Nat.strong_induction_on
      generalizing n env heap entries with
  | h size ih =>
      obtain ⟨afterPrime, primeEntries, primeRun, primeValid, primeCache, primeTrials, primeOther, primeLength⟩ :=
        PrimeCache.result_preserves (quad a b n) last values trials heap trialsAddress cacheAddress
          entries valid trialsFound cacheFound
      have quadratic : Evaluates program env
          (.binOp (.binOp (.binOp (.name "n") .mult (.name "n")) .add
            (.binOp (.name "a") .mult (.name "n"))) .add (.name "b")) heap (num (quad a b n)) heap :=
        Evaluates.binary
          (Evaluates.binary
            (Evaluates.binary (Evaluates.lookup getN) (Evaluates.lookup getN) rfl)
            (Evaluates.binary (Evaluates.lookup getA) (Evaluates.lookup getN) rfl) rfl)
          (Evaluates.lookup getB) rfl
      have tested : Evaluates program (setLocal env "val" (num (quad a b n)))
          (.unaryOp .not (.call (.name "is_prime") [.name "val", .name "trial_primes", .name "cache"]))
          heap (.scalar (.bool (!isPrimeInt (quad a b n)))) afterPrime := by
        apply Evaluates.unary (Evaluates.call
          (.cons (Evaluates.lookup (value := num (quad a b n)) (by simp [setLocal]))
            (.cons (Evaluates.lookup (by simpa [setLocal] using getTrials))
              (.cons (Evaluates.lookup (by simpa [setLocal] using getCache)) .nil))) primeRun)
        simp [HeapCalls.unary, Mutable.truth, scalarTruth, PrimeCache.answer_eq_isPrimeInt]
      by_cases same : n = first a b
      · refine ⟨afterPrime, primeEntries, ?_, primeValid, primeCache, primeTrials, primeOther, primeLength⟩
        apply BlockReturns.while_loop (middle := heap) (flag := true) Evaluates.constant rfl
        apply BlockReturns.assign quadratic
        apply BlockReturns.branch (middle := afterPrime) (flag := true)
          (by simpa [same, first_spec] using tested) rfl
        exact BlockReturns.return_value (Evaluates.lookup (by simpa [setLocal, same] using getN))
      · have smaller : n < first a b := by omega
        obtain ⟨finalHeap, finalEntries, rest, finalValid, finalCache, finalTrials, finalOther, finalLength⟩ :=
          ih (first a b - (n + 1)) (by omega) (n + 1) (by omega)
            (setLocal (setLocal env "val" (num (quad a b n))) "n" (num (n + 1)))
            (by simpa [setLocal] using getA) (by simpa [setLocal] using getB)
            (by simp [setLocal]) (by simpa [setLocal] using getTrials)
            (by simpa [setLocal] using getCache)
            afterPrime primeEntries primeValid primeTrials primeCache rfl
        refine ⟨finalHeap, finalEntries, ?_, finalValid, finalCache, finalTrials,
          fun other different => (finalOther other different).trans (primeOther other different),
          finalLength.trans primeLength⟩
        apply BlockReturns.while_loop (middle := heap) (flag := true) Evaluates.constant rfl
        apply BlockReturns.assign quadratic
        apply BlockReturns.branch (middle := afterPrime) (flag := false)
          (by simpa [before_first a b n smaller] using tested) rfl
        apply BlockReturns.augAssign (middle := afterPrime) (afterOperation := afterPrime) (left := num n) (value := num (n + 1))
          (by simpa [setLocal] using getN) Evaluates.constant
        · simp [inplace, Mutable.binary, scalarBinary, num]
        · exact rest

/-- The optimized unbounded loop terminates exactly at the reference's first nonprime. -/
theorem result_preserves (a b : Int) (last : Nat) (values : List Nat)
    (trials : PrimeCache.TrialPrimes values last) (heap : Mutable.Heap)
    (trialsAddress cacheAddress : Nat) (entries : List (Constant × Mutable.Value))
    (valid : PrimeCache.Valid entries)
    (trialsFound : heap[trialsAddress]? = some (.list (PrimeCache.numbers values)))
    (cacheFound : heap[cacheAddress]? = some (.dict entries)) :
    ∃ finalHeap finalEntries,
      ReturnsValue program "consecutive_prime_length"
        [num a, num b, .ref trialsAddress, .ref cacheAddress]
        heap (num (consecPrimeLen a b)) finalHeap ∧
      PrimeCache.Valid finalEntries ∧
      finalHeap[cacheAddress]? = some (.dict finalEntries) ∧
      finalHeap[trialsAddress]? = some (.list (PrimeCache.numbers values)) ∧
      (∀ other, cacheAddress ≠ other → finalHeap[other]? = heap[other]?) ∧
      finalHeap.length = heap.length := by
  obtain ⟨finalHeap, finalEntries, run, finalValid, finalCache, finalTrials, finalOther, finalLength⟩ :=
    count_loop a b 0 (by omega) last values trials
      (setLocal (argumentEnv
        ((findFunctionIn program.body "consecutive_prime_length").getD ([], [])).1
        [num a, num b, .ref trialsAddress, .ref cacheAddress]) "n" (num 0))
      rfl rfl rfl trialsAddress cacheAddress rfl rfl heap entries valid trialsFound cacheFound
  refine ⟨finalHeap, finalEntries, ?_, finalValid, finalCache, finalTrials, finalOther, finalLength⟩
  rw [consecPrimeLen_eq_first]
  apply BlockReturns.function rfl rfl
  exact BlockReturns.assign Evaluates.constant run

end Consecutive

namespace Search
open ProjectEulerStatements.P27
private abbrev num (n : Int) : Mutable.Value := .scalar (.int n)

structure Best where
  length : Int
  a : Int
  b : Int
  deriving DecidableEq

def admissible (a b : Int) : Prop :=
  (if b ≠ 2 then a % 2 ≠ 0 else a % 2 = 0) ∧ isPrimeInt (1 + a + b) = true
instance (a b : Int) : Decidable (admissible a b) := by unfold admissible; infer_instance

def step (a b : Int) (best : Best) : Best :=
  if admissible a b ∧ best.length < (consecPrimeLen a b : Int)
    then ⟨consecPrimeLen a b, a, b⟩ else best

def inner (b start : Int) : Nat → Best → Best
  | 0, best => best
  | count + 1, best => inner b (start + 1) count (step start b best)

theorem inner_eq_foldl (b start : Int) (count : Nat) (best : Best) :
    inner b start count best =
      (List.range count).foldl (fun best (i : Nat) => step (start + i) b best) best := by
  induction count generalizing start best with
  | zero => rfl
  | succ count ih =>
      rw [inner, ih, List.range_succ_eq_map, List.foldl_cons]
      simp only [Nat.cast_zero, add_zero, List.foldl_map]
      apply congrArg (fun f : Best → Nat → Best => (List.range count).foldl f (step start b best))
      funext previous index
      congr 1
      push_cast
      omega

def search (values : List Int) : Best :=
  values.foldl (fun best b => inner b (-999) 1999 best) ⟨-1, 0, 0⟩

/-- The source search retains its three shared references while rebinding scalars. -/
structure Locals (env : Mutable.Env) (best : Best) (trialsAddress cacheAddress candidatesAddress : Nat) : Prop where
  bestLen : env "best_len" = some (num best.length)
  bestA : env "best_a" = some (num best.a)
  bestB : env "best_b" = some (num best.b)
  trials : env "trial_primes" = some (.ref trialsAddress)
  cache : env "cache" = some (.ref cacheAddress)
  candidates : env "b_candidates" = some (.ref candidatesAddress)

def HeapValid (heap : Mutable.Heap) (trialsAddress cacheAddress : Nat) (values : List Nat) : Prop :=
  ∃ entries, PrimeCache.Valid entries ∧ heap[cacheAddress]? = some (.dict entries) ∧
    heap[trialsAddress]? = some (.list (PrimeCache.numbers values))

def Preserves (before after : Mutable.Heap) (cacheAddress : Nat) : Prop :=
  (∀ other, cacheAddress ≠ other → after[other]? = before[other]?) ∧ after.length = before.length

theorem Preserves.refl (heap : Mutable.Heap) (cacheAddress : Nat) : Preserves heap heap cacheAddress :=
  ⟨fun _ _ => rfl, rfl⟩

theorem Preserves.trans (first : Preserves before middle cacheAddress)
    (second : Preserves middle after cacheAddress) : Preserves before after cacheAddress :=
  ⟨fun other different => (second.1 other different).trans (first.1 other different), second.2.trans first.2⟩

private theorem prime_call (n : Int) (last : Nat) (values : List Nat)
    (trials : PrimeCache.TrialPrimes values last) (heap : Mutable.Heap)
    (trialsAddress cacheAddress : Nat) (valid : HeapValid heap trialsAddress cacheAddress values) :
    ∃ finalHeap, ReturnsValue program "is_prime" [num n, .ref trialsAddress, .ref cacheAddress]
      heap (.scalar (.bool (isPrimeInt n))) finalHeap ∧
      HeapValid finalHeap trialsAddress cacheAddress values ∧ Preserves heap finalHeap cacheAddress := by
  obtain ⟨entries, entriesValid, cacheFound, trialsFound⟩ := valid
  obtain ⟨finalHeap, finalEntries, run, finalValid, finalCache, finalTrials, other, length⟩ :=
    PrimeCache.result_preserves n last values trials heap trialsAddress cacheAddress entries entriesValid trialsFound cacheFound
  exact ⟨finalHeap, by simpa [PrimeCache.truth, PrimeCache.answer_eq_isPrimeInt] using run,
    ⟨finalEntries, finalValid, finalCache, finalTrials⟩, other, length⟩

private theorem length_call (a b : Int) (last : Nat) (values : List Nat)
    (trials : PrimeCache.TrialPrimes values last) (heap : Mutable.Heap)
    (trialsAddress cacheAddress : Nat) (valid : HeapValid heap trialsAddress cacheAddress values) :
    ∃ finalHeap, ReturnsValue program "consecutive_prime_length"
      [num a, num b, .ref trialsAddress, .ref cacheAddress] heap (num (consecPrimeLen a b)) finalHeap ∧
      HeapValid finalHeap trialsAddress cacheAddress values ∧ Preserves heap finalHeap cacheAddress := by
  obtain ⟨entries, entriesValid, cacheFound, trialsFound⟩ := valid
  obtain ⟨finalHeap, finalEntries, run, finalValid, finalCache, finalTrials, other, length⟩ :=
    Consecutive.result_preserves a b last values trials heap trialsAddress cacheAddress entries entriesValid trialsFound cacheFound
  exact ⟨finalHeap, run, ⟨finalEntries, finalValid, finalCache, finalTrials⟩, other, length⟩

private def outerBody : List Stmt :=
  let statements := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
  match statements[6]? with
  | some (Stmt.forIn _ _ body _) => body
  | _ => []

private def innerBody : List Stmt :=
  match outerBody[0]? with
  | some (Stmt.forIn _ _ body _) => body
  | _ => []

/-- Branch flattening retains the pending loop continuation. -/
private theorem branch_tail
    (evaluated : Evaluates program env condition heap value middle)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : BlockContinues program env ((if flag then yes else no) ++ body) middle finalEnv finalHeap) :
    BlockContinues program env (.ifThen condition yes no :: body) heap finalEnv finalHeap := by
  intro following anchors stack
  exact .next rfl ((evaluated anchors _).trans
    (.next (following := ⟨.block (((if flag then yes else no) ++ body) ++ following) env, stack, middle⟩)
      (by simp [HeapCalls.step, run, truth, List.append_assoc]) (rest following anchors stack)))

private theorem continue_tail
    (target : ∀ following, Calls.loopTarget (body ++ following) = some (next, following))
    (rest : BlockContinues program env [next] heap finalEnv finalHeap) :
    BlockContinues program env (.continueLoop :: body) heap finalEnv finalHeap := by
  intro following anchors stack
  exact .next (by simp [HeapCalls.step, target]) (rest following anchors stack)

private theorem Locals.setOther (locals : Locals env best trialsAddress cacheAddress candidatesAddress)
    (name : String) (value : Mutable.Value)
    (other : name ∉ ["best_len", "best_a", "best_b", "trial_primes", "cache", "b_candidates"]) :
    Locals (setLocal env name value) best trialsAddress cacheAddress candidatesAddress := by
  rcases locals with ⟨hl, ha, hb, ht, hc, hp⟩
  simp only [List.mem_cons, not_or] at other
  constructor <;> simp_all [setLocal, eq_comm]

private def improvedEnv (env : Mutable.Env) (a b : Int) :=
  setLocal (setLocal (setLocal env "best_len" (num (consecPrimeLen a b))) "best_a" (num a)) "best_b" (num b)

private theorem Locals.improved (locals : Locals env best trialsAddress cacheAddress candidatesAddress) (a b : Int) :
    Locals (improvedEnv env a b) ⟨consecPrimeLen a b, a, b⟩ trialsAddress cacheAddress candidatesAddress := by
  rcases locals with ⟨hl, ha, hb, ht, hc, hp⟩
  constructor <;> simp_all [improvedEnv, setLocal]

private theorem parity_body (a b : Int) (env : Mutable.Env) (heap : Mutable.Heap)
    (getA : env "a" = some (num a)) (getB : env "b" = some (num b)) (next : Stmt)
    (skipped : ¬ (if b ≠ 2 then a % 2 ≠ 0 else a % 2 = 0) →
      BlockContinues program env [next] heap finalEnv finalHeap)
    (entered : (if b ≠ 2 then a % 2 ≠ 0 else a % 2 = 0) →
      BlockContinues program env (innerBody.drop 1 ++ [.loopResume next]) heap finalEnv finalHeap) :
    BlockContinues program env (innerBody ++ [.loopResume next]) heap finalEnv finalHeap := by
  have btest : Evaluates program env (.compare (.name "b") [(.notEq, .constant (.int 2))])
      heap (.scalar (.bool (decide (b ≠ 2)))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap) (Evaluates.lookup getB) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have atest (op : CmpOp) : Evaluates program env
      (.compare (.binOp (.name "a") .mod (.constant (.int 2))) [(op, .constant (.int 0))])
      heap (.scalar (.bool (evalCmpOp op (a % 2) 0))) heap := by
    apply Evaluates.compare (middle := heap) (afterRight := heap)
      (av := num (a % 2)) (Evaluates.binary (Evaluates.lookup getA) Evaluates.constant rfl) Evaluates.constant
    cases op <;> simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases two : b = 2
  · apply branch_tail (middle := heap) (flag := false) (by simpa [two] using btest) rfl
    by_cases even : a % 2 = 0
    · apply branch_tail (middle := heap) (flag := false) (by simpa [even, evalCmpOp] using atest .notEq) rfl
      exact entered (by simpa [two] using even)
    · apply branch_tail (middle := heap) (flag := true) (by simpa [even, evalCmpOp] using atest .notEq) rfl
      exact continue_tail (by intro following; rfl) (skipped (by simpa [two] using even))
  · apply branch_tail (middle := heap) (flag := true) (by simpa [two] using btest) rfl
    by_cases even : a % 2 = 0
    · apply branch_tail (middle := heap) (flag := true) (by simpa [even, evalCmpOp] using atest .eq) rfl
      exact continue_tail (by intro following; rfl) (skipped (by simpa [two] using even))
    · apply branch_tail (middle := heap) (flag := false) (by simpa [even, evalCmpOp] using atest .eq) rfl
      exact entered (by simpa [two] using even)

private theorem improve_body (a b : Int) (best : Best) (env : Mutable.Env) (heap : Mutable.Heap)
    (getA : env "a" = some (num a)) (getB : env "b" = some (num b))
    (getLength : env "length" = some (num (consecPrimeLen a b)))
    (getBest : env "best_len" = some (num best.length)) :
    BlockContinues program env (innerBody.drop 3) heap
      (if best.length < (consecPrimeLen a b : Int) then improvedEnv env a b else env) heap := by
  have tested : Evaluates program env (.compare (.name "length") [(.gt, .name "best_len")])
      heap (.scalar (.bool (decide (best.length < (consecPrimeLen a b : Int))))) heap := by
    exact Evaluates.compare (Evaluates.lookup getLength) (Evaluates.lookup getBest)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases improve : best.length < (consecPrimeLen a b : Int)
  · simp only [improve, if_true]
    apply BlockContinues.branch (middle := heap) (flag := true) (by simpa [improve] using tested) rfl
    apply BlockContinues.append (BlockContinues.assign (Evaluates.lookup getLength))
    apply BlockContinues.append (BlockContinues.assign (Evaluates.lookup (by simpa [setLocal] using getA)))
    exact BlockContinues.assign (Evaluates.lookup (by simpa [setLocal] using getB))
  · simp only [improve, if_false]
    exact BlockContinues.branch (middle := heap) (flag := false) (by simpa [improve] using tested) rfl BlockContinues.nil

private theorem inner_loop (b start stop : Int) (count : Nat) (stopEq : stop = start + count)
    (best : Best) (last : Nat) (values : List Nat) (trials : PrimeCache.TrialPrimes values last)
    (env : Mutable.Env) (heap : Mutable.Heap) (trialsAddress cacheAddress candidatesAddress : Nat)
    (locals : Locals env best trialsAddress cacheAddress candidatesAddress)
    (getB : env "b" = some (num b)) (valid : HeapValid heap trialsAddress cacheAddress values) :
    ∃ finalEnv finalHeap,
      BlockContinues program env [.rangeStepLoop "a" start stop 1 innerBody []] heap finalEnv finalHeap ∧
      Locals finalEnv (inner b start count best) trialsAddress cacheAddress candidatesAddress ∧
      HeapValid finalHeap trialsAddress cacheAddress values ∧ Preserves heap finalHeap cacheAddress := by
  induction count generalizing start best env heap with
  | zero =>
      refine ⟨env, heap, BlockContinues.range_done ?_ BlockContinues.nil, locals, valid, Preserves.refl heap _⟩
      simp [Calls.rangeContinues, stopEq]
  | succ count ih =>
      let current := setLocal env "a" (num start)
      have currentLocals := locals.setOther "a" (num start) (by decide)
      have currentB : current "b" = some (num b) := by simpa [current, setLocal] using getB
      have running : Calls.rangeContinues start stop 1 = true := by simp [Calls.rangeContinues, stopEq]
      let next := Stmt.rangeStepLoop "a" (start + 1) stop 1 innerBody []
      by_cases parity : if b ≠ 2 then start % 2 ≠ 0 else start % 2 = 0
      · obtain ⟨afterPrime, primeRun, primeValid, primePreserves⟩ :=
          prime_call (1 + start + b) last values trials heap trialsAddress cacheAddress valid
        have quick : Evaluates program current
            (.unaryOp .not (.call (.name "is_prime")
              [.binOp (.binOp (.constant (.int 1)) .add (.name "a")) .add (.name "b"),
                .name "trial_primes", .name "cache"])) heap
            (.scalar (.bool (!isPrimeInt (1 + start + b)))) afterPrime := by
          apply Evaluates.unary (Evaluates.call
            (.cons (Evaluates.binary (Evaluates.binary Evaluates.constant (Evaluates.lookup rfl) rfl)
              (Evaluates.lookup currentB) rfl)
              (.cons (Evaluates.lookup currentLocals.trials) (.cons (Evaluates.lookup currentLocals.cache) .nil))) primeRun)
          simp [HeapCalls.unary, Mutable.truth, scalarTruth]
        by_cases prime : isPrimeInt (1 + start + b) = true
        · obtain ⟨afterLength, lengthRun, lengthValid, lengthPreserves⟩ :=
            length_call start b last values trials afterPrime trialsAddress cacheAddress primeValid
          let withLength := setLocal current "length" (num (consecPrimeLen start b))
          let updated := if best.length < (consecPrimeLen start b : Int) then improvedEnv withLength start b else withLength
          have withLengthLocals := currentLocals.setOther "length" (num (consecPrimeLen start b)) (by decide)
          have updatedLocals : Locals updated (step start b best) trialsAddress cacheAddress candidatesAddress := by
            dsimp [updated, step]
            simp only [admissible, parity, prime, and_self, true_and]
            split
            · exact withLengthLocals.improved start b
            · exact withLengthLocals
          have updatedB : updated "b" = some (num b) := by
            dsimp [updated, improvedEnv, withLength]
            split <;> simpa [setLocal] using currentB
          obtain ⟨finalEnv, finalHeap, rest, finalLocals, finalValid, finalPreserves⟩ :=
            ih (start + 1) (by simp only [Nat.cast_add, Nat.cast_one] at stopEq; omega)
              (step start b best) updated afterLength updatedLocals updatedB lengthValid
          refine ⟨finalEnv, finalHeap, ?_, finalLocals, finalValid,
            primePreserves.trans (lengthPreserves.trans finalPreserves)⟩
          apply BlockContinues.range_step running
          apply parity_body start b current heap rfl currentB next
          · exact fun impossible => False.elim (impossible parity)
          · intro _
            apply branch_tail (middle := afterPrime) (flag := false) (by simpa [prime] using quick) rfl
            apply BlockContinues.append (BlockContinues.assign (Evaluates.call
              (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup currentB)
                (.cons (Evaluates.lookup currentLocals.trials) (.cons (Evaluates.lookup currentLocals.cache) .nil)))) lengthRun))
            apply BlockContinues.append (improve_body start b best withLength afterLength
              (by simp [withLength, current, setLocal]) (by simpa [withLength, setLocal] using currentB)
              rfl withLengthLocals.bestLen)
            exact BlockContinues.loop_resume rest
        · have unchanged : step start b best = best := by simp [step, admissible, prime]
          obtain ⟨finalEnv, finalHeap, rest, finalLocals, finalValid, finalPreserves⟩ :=
            ih (start + 1) (by simp only [Nat.cast_add, Nat.cast_one] at stopEq; omega)
              best current afterPrime currentLocals currentB primeValid
          refine ⟨finalEnv, finalHeap, ?_, ?_, finalValid, primePreserves.trans finalPreserves⟩
          · apply BlockContinues.range_step running
            apply parity_body start b current heap rfl currentB next
            · exact fun impossible => False.elim (impossible parity)
            · intro _
              apply branch_tail (middle := afterPrime) (flag := true)
                (by simpa [Bool.eq_false_iff.mpr prime] using quick) rfl
              exact continue_tail (by intro following; rfl) rest
          · simpa only [inner, unchanged] using finalLocals
      · have unchanged : step start b best = best := by simp only [step, admissible, parity, false_and, if_false]
        obtain ⟨finalEnv, finalHeap, rest, finalLocals, finalValid, finalPreserves⟩ :=
          ih (start + 1) (by simp only [Nat.cast_add, Nat.cast_one] at stopEq; omega)
            best current heap currentLocals currentB valid
        refine ⟨finalEnv, finalHeap, ?_, ?_, finalValid, finalPreserves⟩
        · apply BlockContinues.range_step running
          exact parity_body start b current heap rfl currentB next (fun _ => rest)
            (fun impossible => False.elim (parity impossible))
        · simpa only [inner, unchanged] using finalLocals

abbrev numbers (values : List Int) : List Mutable.Value := values.map num

private theorem outer_loop (candidates processed remaining : List Int) (split : candidates = processed ++ remaining)
    (best : Best) (last : Nat) (values : List Nat) (trials : PrimeCache.TrialPrimes values last)
    (env : Mutable.Env) (heap : Mutable.Heap) (trialsAddress cacheAddress candidatesAddress : Nat)
    (locals : Locals env best trialsAddress cacheAddress candidatesAddress)
    (valid : HeapValid heap trialsAddress cacheAddress values)
    (candidatesFound : heap[candidatesAddress]? = some (.list (numbers candidates))) :
    ∃ finalEnv finalHeap,
      BlockContinues program env [.sequenceStepLoop "b" (.ref candidatesAddress) processed.length outerBody []]
        heap finalEnv finalHeap ∧
      Locals finalEnv (remaining.foldl (fun best b => inner b (-999) 1999 best) best)
        trialsAddress cacheAddress candidatesAddress ∧
      HeapValid finalHeap trialsAddress cacheAddress values ∧ Preserves heap finalHeap cacheAddress := by
  induction remaining generalizing processed best env heap with
  | nil =>
      have same : candidates = processed := by simpa using split
      refine ⟨env, heap, ?_, locals, valid, Preserves.refl heap _⟩
      exact BlockContinues.sequence_done (items := numbers candidates)
        (by simp [sequenceValue, Mutable.sequence, candidatesFound]) (by simp [numbers, same]) BlockContinues.nil
  | cons b remaining ih =>
      let current := setLocal env "b" (num b)
      have currentLocals := locals.setOther "b" (num b) (by decide)
      obtain ⟨afterInner, innerHeap, innerRun, innerLocals, innerValid, innerPreserves⟩ :=
        inner_loop b (-999) 1000 1999 (by norm_num) best last values trials current heap
          trialsAddress cacheAddress candidatesAddress currentLocals rfl valid
      have different : cacheAddress ≠ candidatesAddress := by
        obtain ⟨entries, _, cacheFound, _⟩ := valid
        intro same
        rw [same, candidatesFound] at cacheFound
        simp at cacheFound
      have candidatesKept : innerHeap[candidatesAddress]? = some (.list (numbers candidates)) := by
        rw [innerPreserves.1 candidatesAddress different]
        exact candidatesFound
      obtain ⟨finalEnv, finalHeap, rest, finalLocals, finalValid, finalPreserves⟩ :=
        ih (processed ++ [b]) (by simpa [List.append_assoc] using split)
          (inner b (-999) 1999 best) afterInner innerHeap innerLocals innerValid candidatesKept
      refine ⟨finalEnv, finalHeap, ?_, ?_, finalValid, innerPreserves.trans finalPreserves⟩
      · apply BlockContinues.sequence_step (items := numbers candidates) (value := num b)
          (by simp [sequenceValue, Mutable.sequence, candidatesFound]) (by simp [numbers, split])
        apply BlockContinues.append (BlockContinues.for_range
          (.cons (Evaluates.unary Evaluates.constant rfl) (.cons Evaluates.constant .nil)) rfl innerRun)
        simpa only [List.length_append, List.length_singleton] using rest
      · simpa only [List.foldl_cons] using finalLocals

/-- The two source loops implement the b-outer, a-inner fold, including pruning and strict ties. -/
theorem loops (candidates : List Int) (best : Best) (last : Nat) (values : List Nat)
    (trials : PrimeCache.TrialPrimes values last) (env : Mutable.Env) (heap : Mutable.Heap)
    (trialsAddress cacheAddress candidatesAddress : Nat)
    (locals : Locals env best trialsAddress cacheAddress candidatesAddress)
    (valid : HeapValid heap trialsAddress cacheAddress values)
    (candidatesFound : heap[candidatesAddress]? = some (.list (numbers candidates))) :
    ∃ finalEnv finalHeap,
      BlockContinues program env
        ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 6 |>.take 1)
        heap finalEnv finalHeap ∧
      Locals finalEnv (candidates.foldl (fun best b => inner b (-999) 1999 best) best)
        trialsAddress cacheAddress candidatesAddress ∧
      HeapValid finalHeap trialsAddress cacheAddress values ∧ Preserves heap finalHeap cacheAddress := by
  obtain ⟨finalEnv, finalHeap, run, finalLocals, finalValid, finalPreserves⟩ :=
    outer_loop candidates [] candidates rfl best last values trials env heap trialsAddress cacheAddress
      candidatesAddress locals valid candidatesFound
  refine ⟨finalEnv, finalHeap, ?_, finalLocals, finalValid, finalPreserves⟩
  exact BlockContinues.for_sequence (items := numbers candidates) (middle := heap)
    (by intro expressions; simp) (Evaluates.lookup locals.candidates)
    (by simp [Mutable.sequence, candidatesFound]) run

/-- Once initialized, `solve` returns the product selected by its mathematical search fold. -/
theorem result (candidates : List Int) (last : Nat) (values : List Nat)
    (trials : PrimeCache.TrialPrimes values last) (env : Mutable.Env) (heap : Mutable.Heap)
    (trialsAddress cacheAddress candidatesAddress : Nat)
    (locals : Locals env ⟨-1, 0, 0⟩ trialsAddress cacheAddress candidatesAddress)
    (valid : HeapValid heap trialsAddress cacheAddress values)
    (candidatesFound : heap[candidatesAddress]? = some (.list (numbers candidates))) :
    ∃ finalHeap,
      BlockReturns program env
        ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 6)
        heap (num ((search candidates).a * (search candidates).b)) finalHeap ∧
      HeapValid finalHeap trialsAddress cacheAddress values ∧ Preserves heap finalHeap cacheAddress := by
  obtain ⟨finalEnv, finalHeap, run, finalLocals, finalValid, finalPreserves⟩ :=
    loops candidates ⟨-1, 0, 0⟩ last values trials env heap trialsAddress cacheAddress candidatesAddress
      locals valid candidatesFound
  refine ⟨finalHeap, ?_, finalValid, finalPreserves⟩
  apply BlockContinues.returns run
  exact BlockReturns.return_value (Evaluates.binary
    (Evaluates.lookup finalLocals.bestA) (Evaluates.lookup finalLocals.bestB) rfl)

/-- Candidate order made explicit for connecting the source fold to a mathematical winner. -/
def pairs (candidates : List Int) : List (Int × Int) :=
  candidates.flatMap fun b => (List.range 1999).map fun (i : Nat) => ((-999 : Int) + i, b)

theorem search_eq_fold (candidates : List Int) :
    search candidates = (pairs candidates).foldl (fun best pair => step pair.1 pair.2 best) ⟨-1, 0, 0⟩ := by
  simp only [search, pairs, List.foldl_flatMap, List.foldl_map, inner_eq_foldl]

/-- An already selected winner or a strictly shorter candidate. -/
def Below (target best : Best) : Prop := best = target ∨ best.length < target.length

/-- Every admitted candidate is shorter than the target, except the target itself. -/
def Dominates (target : Best) (pair : Int × Int) : Prop :=
  admissible pair.1 pair.2 →
    (consecPrimeLen pair.1 pair.2 : Int) ≤ target.length ∧
    ((consecPrimeLen pair.1 pair.2 : Int) = target.length → pair = (target.a, target.b))

private theorem step_below (target best : Best) (a b : Int) (before : Below target best)
    (dominates : Dominates target (a, b)) : Below target (step a b best) := by
  by_cases improve : admissible a b ∧ best.length < (consecPrimeLen a b : Int)
  · rw [step, if_pos improve]
    have bounds := dominates improve.1
    dsimp at bounds
    by_cases full : (consecPrimeLen a b : Int) = target.length
    · left
      have same := bounds.2 full
      cases target
      simp_all
    · right
      change (consecPrimeLen a b : Int) < target.length
      omega
  · simpa only [step, if_neg improve] using before

private theorem step_stable (target : Best) (a b : Int) (dominates : Dominates target (a, b)) :
    step a b target = target := by
  have cannot : ¬ (admissible a b ∧ target.length < (consecPrimeLen a b : Int)) := by
    intro improve
    have bound := (dominates improve.1).1
    change (consecPrimeLen a b : Int) ≤ target.length at bound
    omega
  simp only [step, if_neg cannot]

private theorem step_winner (target best : Best) (before : Below target best)
    (admitted : admissible target.a target.b)
    (score : target.length = (consecPrimeLen target.a target.b : Int)) :
    step target.a target.b best = target := by
  rcases before with rfl | shorter
  · simp [step, ← score]
  · simp only [step, admitted, true_and, ← score, if_pos shorter]

private theorem fold_stable (target : Best) (values : List (Int × Int))
    (dominates : ∀ pair ∈ values, Dominates target pair) :
    values.foldl (fun best pair => step pair.1 pair.2 best) target = target := by
  induction values with
  | nil => rfl
  | cons pair rest ih =>
      rw [List.foldl_cons, step_stable target pair.1 pair.2 (dominates pair (by simp))]
      exact ih (fun pair member => dominates pair (by simp [member]))

/-- Strict updates find a unique maximum independently of the candidate list's order. -/
theorem fold_winner (target best : Best) (values : List (Int × Int))
    (before : Below target best) (admitted : admissible target.a target.b)
    (score : target.length = (consecPrimeLen target.a target.b : Int))
    (dominates : ∀ pair ∈ values, Dominates target pair)
    (present : (target.a, target.b) ∈ values) :
    values.foldl (fun best pair => step pair.1 pair.2 best) best = target := by
  induction values generalizing best with
  | nil => simp at present
  | cons pair rest ih =>
      rcases List.mem_cons.mp present with same | later
      · subst pair
        rw [List.foldl_cons, step_winner target best before admitted score]
        exact fold_stable target rest (fun pair member => dominates pair (by simp [member]))
      · rw [List.foldl_cons]
        exact ih (step pair.1 pair.2 best)
          (step_below target best pair.1 pair.2 before (dominates pair (by simp)))
          (fun pair member => dominates pair (by simp [member])) later

theorem mem_pairs (candidates : List Int) (a b : Int) :
    (a, b) ∈ pairs candidates ↔ b ∈ candidates ∧ -999 ≤ a ∧ a ≤ 999 := by
  constructor
  · intro member
    obtain ⟨b', present, member⟩ := List.mem_flatMap.mp member
    obtain ⟨i, inside, equal⟩ := List.mem_map.mp member
    have bound := List.mem_range.mp inside
    have value : (-999 : Int) + i = a := congrArg Prod.fst equal
    have same : b' = b := congrArg Prod.snd equal
    exact ⟨same ▸ present, by omega, by omega⟩
  · rintro ⟨present, lower, upper⟩
    apply List.mem_flatMap.mpr
    refine ⟨b, present, List.mem_map.mpr ⟨(a + 999).toNat, List.mem_range.mpr (by omega), ?_⟩⟩
    apply Prod.ext
    · change (-999 : Int) + (a + 999).toNat = a
      omega
    · rfl

/-- A unique mathematical winner determines the result of the optimized source search. -/
theorem search_winner (candidates : List Int) (target : Best)
    (present : target.b ∈ candidates) (lower : -999 ≤ target.a) (upper : target.a ≤ 999)
    (positive : -1 < target.length) (admitted : admissible target.a target.b)
    (score : target.length = (consecPrimeLen target.a target.b : Int))
    (dominates : ∀ a b, b ∈ candidates → -999 ≤ a → a ≤ 999 → Dominates target (a, b)) :
    search candidates = target := by
  rw [search_eq_fold]
  apply fold_winner target ⟨-1, 0, 0⟩ (pairs candidates) (Or.inr positive) admitted score
  · rintro ⟨a, b⟩ member
    obtain ⟨present, lower, upper⟩ := (mem_pairs candidates a b).mp member
    exact dominates a b present lower upper
  · exact (mem_pairs candidates target.a target.b).mpr ⟨present, lower, upper⟩

end Search

namespace Initialize
private abbrev num (n : Int) : Mutable.Value := .scalar (.int n)

/-- Sorted complete prime prefixes satisfy trial division without enumerating their final prime. -/
theorem trial_primes (limit : Nat) (large : 3 ≤ limit) :
    ∃ last, PrimeCache.TrialPrimes (Sieve.primes limit) last := by
  have three : 3 ∈ Sieve.primes limit := (Sieve.mem_primes limit 3 (by omega)).mpr ⟨by decide, large⟩
  have nonempty := List.ne_nil_of_mem three
  let last := (Sieve.primes limit).getLast nonempty
  have lastValue : (Sieve.primes limit).getLast? = some last := List.getLast?_eq_some_getLast nonempty
  have lastMember := List.mem_of_getLast? lastValue
  have bound : ∀ p ∈ Sieve.primes limit, p ≤ last := by
    obtain ⟨front, split⟩ := List.getLast?_eq_some_iff.mp lastValue
    have ordered := Sieve.primes_pairwise limit
    rw [split] at ordered
    intro p member
    rw [split, List.mem_append] at member
    rcases member with earlier | final
    · exact ((List.pairwise_append.mp ordered).2.2 p earlier last (by simp)).le
    · exact (List.mem_singleton.mp final).le
  have largeLast := bound 3 three
  have lastPrime := ((Sieve.mem_primes limit last (by omega)).mp lastMember).1
  have lastBound := ((Sieve.mem_primes limit last (by omega)).mp lastMember).2
  refine ⟨last, Sieve.primes_pairwise limit, ?_, bound, lastValue, largeLast, ?_, ?_⟩
  · intro p member
    exact ((Sieve.mem_primes limit p (by omega)).mp member).1
  · exact lastPrime.eq_two_or_odd.resolve_left (by omega)
  · intro p prime before
    exact (Sieve.mem_primes limit p (by omega)).mpr ⟨prime, before.trans lastBound⟩

private def copyBody (collector : Nat) : List Stmt :=
  [.ifThen (.compare (.name "p") [(.ltE, .constant (.int 1000))])
    [.comprehensionAppend collector (.name "p")] []]

private theorem copy_loop (heap : Mutable.Heap) (address : Nat) (items remaining : List Nat)
    (bounded : ∀ p ∈ items, p ≤ 1000)
    (found : heap[address]? = some (.list (Sieve.numbers items))) (position : Nat)
    (suffix : items.drop position = remaining) (env : Mutable.Env) (done : List Nat) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "p" (.ref address) position (copyBody heap.length) []]
      (heap ++ [.list (Sieve.numbers done)]) finalEnv (heap ++ [.list (Sieve.numbers (done ++ remaining))]) := by
  have inHeap : address < heap.length := List.getElem?_eq_some_iff.mp found |>.1
  have sequence : Mutable.sequence (.ref address) (heap ++ [.list (Sieve.numbers done)]) =
      some (.ok (Sieve.numbers items), heap ++ [.list (Sieve.numbers done)]) := by
    simp [Mutable.sequence, List.getElem?_append_left inHeap, found]
  induction remaining generalizing position env done with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.append_nil]
      apply BlockContinues.sequence_done (source := .ref address) sequence
      · have absent : items[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [Sieve.numbers, absent]
      · exact BlockContinues.nil
  | cons head tail ih =>
      let bound := setLocal env "p" (num head)
      have item : items[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      have headBound := bounded head (List.mem_of_getElem? item)
      have emitted := BlockContinues.comprehension_append (globals := [])
        (module := program) (env := bound) (address := heap.length)
        (heap := heap ++ [.list (Sieve.numbers done)])
        (Evaluates.lookup (name := "p") rfl)
        (by simp : (heap ++ [.list (Sieve.numbers done)])[heap.length]? = some (.list (Sieve.numbers done)))
      have step : BlockContinues program bound (copyBody heap.length)
          (heap ++ [.list (Sieve.numbers done)]) bound (heap ++ [.list (Sieve.numbers (done ++ [head]))]) := by
        apply BlockContinues.branch (middle := heap ++ [.list (Sieve.numbers done)]) (flag := true)
          (value := .scalar (.bool true))
        · apply Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
          simp [HeapCalls.compare, scalarCompare, evalCmpOp]
          omega
        · rfl
        · simpa [Sieve.numbers, List.map_append] using emitted
      obtain ⟨finalEnv, rest⟩ := ih (position + 1)
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) bound (done ++ [head])
        (by simp [Mutable.sequence, List.getElem?_append_left inHeap, found])
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (source := .ref address) (value := num head) sequence
        (by simp [Sieve.numbers, item])
      simpa [copyBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, List.append_assoc] using step.append rest

private theorem candidates (env : Mutable.Env) (heap : Mutable.Heap) :
    ∃ bytes, Evaluates program env
      (.listComp (.name "p") [("p", .call (.name "sieve_primes") [.constant (.int 1000)],
        [.compare (.name "p") [(.ltE, .constant (.int 1000))]])])
      heap (.ref (heap.length + 3))
      (heap ++ [.bytearray [1], .bytearray bytes, .list (Sieve.numbers (Sieve.primes 1000)),
        .list (Sieve.numbers (Sieve.primes 1000))]) := by
  obtain ⟨bytes, sieved⟩ := Sieve.result 1000 (by omega) heap
  let middle := heap ++ [Mutable.Object.bytearray [1], .bytearray bytes, .list (Sieve.numbers (Sieve.primes 1000))]
  have found : middle[heap.length + 2]? = some (.list (Sieve.numbers (Sieve.primes 1000))) := by simp [middle]
  obtain ⟨finalEnv, copied⟩ := copy_loop middle (heap.length + 2) (Sieve.primes 1000) (Sieve.primes 1000)
    (fun p member => ((Sieve.mem_primes 1000 p (by omega)).mp member).2) found 0 rfl
    (comprehensionEnv env [("p", .call (.name "sieve_primes") [.constant (.int 1000)],
      [.compare (.name "p") [(.ltE, .constant (.int 1000))]])]) []
  refine ⟨bytes, ?_⟩
  have result := Evaluates.list_comp_sequence (module := program) (env := env)
    (items := Sieve.numbers (Sieve.primes 1000)) (middle := middle)
    (by intro expressions; simp)
    (Evaluates.call (.cons Evaluates.constant .nil) sieved)
    (by simp [Mutable.sequence, found]) copied
  simpa [middle, copyBody, comprehensionFilters, comprehensionBody, List.append_assoc] using result

def candidateValues : List Int := (Sieve.primes 1000).map Int.ofNat

def optimizedAnswer : Int :=
  (Search.search candidateValues).a * (Search.search candidateValues).b

/-- Execute both sieves, the candidate comprehension, cache allocation, and the optimized search. -/
theorem result :
    ∃ finalHeap, ReturnsValue program "solve" [] [] (num optimizedAnswer) finalHeap := by
  obtain ⟨last, trials⟩ := trial_primes 100000 (by omega)
  obtain ⟨firstBytes, sieved⟩ := Sieve.result 100000 (by omega) []
  let firstHeap : Mutable.Heap :=
    [.bytearray [1], .bytearray firstBytes, .list (Sieve.numbers (Sieve.primes 100000))]
  let withTrials := setLocal (argumentEnv [] []) "trial_primes" (.ref 2)
  obtain ⟨secondBytes, copied⟩ := candidates withTrials firstHeap
  let beforeCache : Mutable.Heap := firstHeap ++
    [.bytearray [1], .bytearray secondBytes, .list (Sieve.numbers (Sieve.primes 1000)),
      .list (Sieve.numbers (Sieve.primes 1000))]
  let allocated := beforeCache ++ [Mutable.Object.dict []]
  let ready := setLocal (setLocal (setLocal (setLocal (setLocal withTrials "b_candidates" (.ref 6))
    "cache" (.ref 7)) "best_len" (num (-1))) "best_a" (num 0)) "best_b" (num 0)
  have locals : Search.Locals ready ⟨-1, 0, 0⟩ 2 7 6 := ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  have valid : Search.HeapValid allocated 2 7 (Sieve.primes 100000) := by
    refine ⟨[], PrimeCache.valid_empty, ?_, ?_⟩
    · simp [allocated, beforeCache, firstHeap]
    · simp [allocated, beforeCache, firstHeap, Sieve.numbers, PrimeCache.numbers]
  have found : allocated[6]? = some (.list (Search.numbers candidateValues)) := by
    simp [allocated, beforeCache, firstHeap, candidateValues, Search.numbers, Sieve.numbers, List.map_map]
  obtain ⟨finalHeap, searched, _, _⟩ :=
    Search.result candidateValues last (Sieve.primes 100000) trials ready allocated 2 7 6 locals valid found
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (middle := firstHeap) (value := .ref 2)
    (Evaluates.call (.cons Evaluates.constant .nil) sieved)
  apply BlockReturns.assign (middle := beforeCache) (value := .ref 6)
  · simpa [firstHeap, beforeCache] using copied
  · have emptyDict : Evaluates program (setLocal withTrials "b_candidates" (.ref 6)) (.dict [])
        beforeCache (.ref 7) allocated := by
      simpa [allocated, beforeCache, firstHeap] using
        (Evaluates.dictionary (globals := []) (module := program) (env := setLocal withTrials "b_candidates" (.ref 6))
          (heap := beforeCache) .nil)
    apply BlockContinues.returns (BlockContinues.annAssign emptyDict)
    apply BlockReturns.assign (Evaluates.unary Evaluates.constant rfl)
    apply BlockReturns.assign Evaluates.constant
    apply BlockReturns.assign Evaluates.constant
    exact searched

end Initialize

theorem solve.«Returns» : (HeapCalls.scalarFunction program "solve" []).Returns := by
  obtain ⟨heap, run⟩ := Initialize.result
  exact ReturnsValue.scalar_returns (values := []) run

namespace Canonical
open ProjectEulerStatements.P27
private def coefficients (bound : Nat) : List Int :=
  (List.range (2 * bound + 1)).map (fun (i : Nat) => (i : Int) - bound)
private def rectangle (limitA limitB : Nat) : List (Int × Int) :=
  (coefficients limitA).flatMap (fun a => (coefficients limitB).map (a, ·))
private def selectedProduct (best : Option (Int × Int)) : Int :=
  let chosen := best.getD (0, 0)
  chosen.1 * chosen.2
private theorem naive_eq_argmax (limitA limitB : Nat) :
    naive limitA limitB = selectedProduct
      (List.argmax (fun p : Int × Int => consecPrimeLen p.1 p.2) (rectangle limitA limitB)) := by
  simp only [naive, rectangle, coefficients, selectedProduct, bind_pure_comp,
    List.map_eq_map, List.map_map, Function.comp_def, List.foldr_append_eq_append, List.append_nil,
    List.flatMap, Option.getD]
  generalize List.argmax (fun p : Int × Int => consecPrimeLen p.1 p.2) _ = best
  cases best <;> rfl
private theorem mem_coefficients (bound : Nat) (a : Int) :
    a ∈ coefficients bound ↔ -(bound : Int) ≤ a ∧ a ≤ bound := by
  constructor
  · intro member
    obtain ⟨i, inside, equal⟩ := List.mem_map.mp member
    have boundI := List.mem_range.mp inside
    omega
  · rintro ⟨lower, upper⟩
    apply List.mem_map.mpr
    refine ⟨(a + bound).toNat, List.mem_range.mpr (by omega), ?_⟩
    omega
private theorem mem_rectangle (a b : Int) :
    (a, b) ∈ rectangle 999 1000 ↔ -999 ≤ a ∧ a ≤ 999 ∧ -1000 ≤ b ∧ b ≤ 1000 := by
  simp only [rectangle, List.mem_flatMap, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨x, xMem, y, yMem, hx, hy⟩
    subst x
    subst y
    exact ⟨(mem_coefficients 999 a).mp xMem |>.1,
      (mem_coefficients 999 a).mp xMem |>.2,
      (mem_coefficients 1000 b).mp yMem |>.1,
      (mem_coefficients 1000 b).mp yMem |>.2⟩
  · rintro ⟨al, au, bl, bu⟩
    exact ⟨a, (mem_coefficients 999 a).mpr ⟨al,au⟩,
      b, (mem_coefficients 1000 b).mpr ⟨bl,bu⟩, rfl, rfl⟩
theorem reference_winner
    (score : consecPrimeLen (-61) 971 = 71)
    (unique : ∀ a b : Int, a ≤ 999 → b ≤ 1000 →
      71 ≤ consecPrimeLen a b → a = -61 ∧ b = 971)
    (bound : ∀ a b : Int, a ≤ 999 → b ≤ 1000 → consecPrimeLen a b ≤ 71) :
    naive 999 1000 = -59231 := by
  have selected : List.argmax (fun p : Int × Int => consecPrimeLen p.1 p.2) (rectangle 999 1000) = some (-61,971) := by
    apply List.argmax_eq_some_iff.mpr
    refine ⟨(mem_rectangle (-61) 971).mpr (by norm_num), ?_, ?_⟩
    · rintro ⟨a,b⟩ member
      have bounds := (mem_rectangle a b).mp member
      simpa only [score] using bound a b bounds.2.1 bounds.2.2.2
    · rintro ⟨a,b⟩ member atLeast
      have bounds := (mem_rectangle a b).mp member
      obtain ⟨rfl,rfl⟩ := unique a b bounds.2.1 bounds.2.2.2 (by simpa [score] using atLeast)
      exact Nat.le_refl _
  rw [naive_eq_argmax, selected]
  rfl

private def target : Search.Best := ⟨71, -61, 971⟩

theorem optimized_winner : Search.search Initialize.candidateValues = target := by
  apply Search.search_winner Initialize.candidateValues target
  · apply List.mem_map.mpr
    exact ⟨971, (Sieve.mem_primes 1000 971 (by omega)).mpr ⟨by norm_num, by omega⟩, rfl⟩
  · decide
  · decide
  · decide
  · norm_num [target, Search.admissible, isPrimeInt, Int.toNat]
  · simp [target, ProjectEulerStatements.QuadraticPrimeWinner.canonical_length]
  · intro a b member _ upper admitted
    obtain ⟨p, present, rfl⟩ := List.mem_map.mp member
    have upperP := ((Sieve.mem_primes 1000 p (by omega)).mp present).2
    have upperB : (p : Int) ≤ 1000 := by exact_mod_cast upperP
    constructor
    · have bound := ProjectEulerStatements.QuadraticPrimeWinner.score_le a p upper upperB
      change (consecPrimeLen a p : Int) ≤ 71
      exact_mod_cast bound
    · intro same
      have long : 71 ≤ consecPrimeLen a p := by change (consecPrimeLen a p : Int) = 71 at same; omega
      obtain ⟨aValue, bValue⟩ := ProjectEulerStatements.QuadraticPrimeWinner.unique_long a p upper upperB long
      exact Prod.ext aValue bValue

theorem optimized_answer : Initialize.optimizedAnswer = -59231 := by
  rw [Initialize.optimizedAnswer, optimized_winner]
  rfl

theorem reference_answer : ProjectEulerStatements.P27.naive 999 1000 = -59231 :=
  reference_winner ProjectEulerStatements.QuadraticPrimeWinner.canonical_length
    ProjectEulerStatements.QuadraticPrimeWinner.unique_long
    ProjectEulerStatements.QuadraticPrimeWinner.score_le

end Canonical

theorem solve.«Equiv(naive)» :
    (HeapCalls.scalarFunction program "solve" []).Equiv
      (HeapCalls.scalarFunction P27Ast.module "naive_default" []) := by
  obtain ⟨optimizedHeap, optimized⟩ := Initialize.result
  rw [Canonical.optimized_answer] at optimized
  obtain ⟨referenceHeap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p27.naive_default.result []
  rw [Canonical.reference_answer] at reference
  have left := HeapCalls.ReturnsValue.scalar_finishes (values := []) optimized
  have right := HeapCalls.ReturnsValue.scalar_finishes (values := []) reference
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem solve.«Supported» :
    (HeapCalls.scalarFunction program "solve" []).Supported ∧
      (HeapCalls.scalarFunction P27Ast.module "naive_default" []).Supported := by
  constructor
  · exact solve.«Returns».supported
  · exact data.«project-euler-statements».data.py.project_euler_statements.p27.naive_default.«Returns».supported

end solvers.«27»
