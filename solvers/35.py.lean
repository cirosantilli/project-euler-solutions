import ProjectEulerStatements.Lib.CircularPrimeCounting
import data.«project-euler-statements».data.py.project_euler_statements.«p35.py»
import «.formalsw».solvers.«35.py.ast»
import ProjectEulerSolutions.Lib.PrimeSieve
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.List.Infix
import lang.py.FormalSW.SetSumCalls
import lang.py.FormalSW.ScalarSetLemmas
import lang.py.FormalSW.TruthCalls
import ProjectEulerStatements.Lib.DecimalDigits
import ProjectEulerStatements.Lib.CircularPrimeStrings

set_option Elab.async false

namespace solvers.«35»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P35Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

namespace prime_sieve
open ProjectEulerSolutions.PrimeSieve

private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "prime_sieve").getD ([], [])).2
private abbrev loopBody : List Stmt :=
  match (code[5]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []
private abbrev markBody : List Stmt :=
  match (loopBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

private def markEnv (env : HeapCalls.Env) (n p : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "start" (number (p*p : Nat))) "step" (number p))
    "count" (number (((n : Int)-1-p*p)/p+1))

/-- The extended slice changes exactly the multiples selected by the shared sieve invariant. -/
private theorem mark_result (env : HeapCalls.Env) (heap : Mutable.Heap) (address n p : Nat)
    (bytes : List UInt8) (lower : 2 ≤ p) (inside : p*p < n) (size : bytes.length = n)
    (getP : env "p" = some (number p)) (getN : env "n" = some (number n))
    (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    BlockContinues program env markBody heap (markEnv env n p)
      (heap.set address (.bytearray (mark bytes p))) := by
  let e1 := setLocal env "start" (number (p*p : Nat))
  let e2 := setLocal e1 "step" (number p)
  let count : Int := ((n : Int)-1-p*p)/p+1
  have first : BlockContinues program env [markBody[0]'(by decide)] heap e1 heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP)
      (by simp [Mutable.binary, scalarBinary, number])
  have second : BlockContinues program e1 [markBody[1]'(by decide)] heap e2 heap :=
    BlockContinues.assign (Evaluates.lookup (by simpa [e1, setLocal] using getP))
  have third : BlockContinues program e2 [markBody[2]'(by decide)] heap (markEnv env n p) heap := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := number (((n : Int)-1-p*p)/p)) (bv := number 1)
      (middle := heap) (afterRight := heap)
    · apply Evaluates.binary (av := number ((n : Int)-1-p*p)) (bv := number p)
        (middle := heap) (afterRight := heap)
      · apply Evaluates.binary (av := number ((n : Int)-1)) (bv := number (p*p : Nat))
          (middle := heap) (afterRight := heap)
        · exact Evaluates.binary (Evaluates.lookup (by simpa [e2, e1, setLocal] using getN))
            Evaluates.constant (by simp [Mutable.binary, scalarBinary, number])
        · exact Evaluates.lookup (by simp [e2, e1, setLocal])
        · simp [Mutable.binary, scalarBinary, number]
      · exact Evaluates.lookup rfl
      · simp [Mutable.binary, scalarBinary, number, show p ≠ 0 by omega, show 0 < p by omega]
    · exact Evaluates.constant
    · rfl
  have fourth : BlockContinues program (markEnv env n p) [markBody[3]'(by decide)] heap
      (markEnv env n p) (heap.set address (.bytearray (mark bytes p))) := by
    apply BlockContinues.setSlice (value := .scalar (.bytes (List.replicate count.toNat 0)))
      (afterRhs := heap) (afterTarget := heap) (receiver := .ref address)
      (lowerValue := number (p*p : Nat)) (upperValue := number n) (strideValue := number p)
    · apply Evaluates.binary (av := .scalar (.bytes [0])) (bv := number count)
        (middle := heap) (afterRight := heap)
      · exact Evaluates.constant
      · exact Evaluates.lookup rfl
      · simp [Mutable.binary, scalarBinary]
    · exact .cons (Evaluates.lookup (by simpa [markEnv, setLocal] using getBuffer))
        (.cons (Evaluates.lookup (by simp [markEnv, setLocal]))
          (.cons (Evaluates.lookup (by simpa [markEnv, setLocal] using getN))
            (.cons (Evaluates.lookup (by simp [markEnv, setLocal])) .nil)))
    · apply Mutable.setSlice_buffer heap address bytes _ _ _ _ _
        { start := some (p*p), stop := some n, step := some p } found
      · change ((if some (p : Int) = some 0 then Mutable.error "ValueError"
          else (pure { start := some ((p : Int)*p), stop := some (n : Int), step := some (p : Int) } : Mutable.Eval FormalSW.Py.Builtin.Slice.Slice)) heap) = _
        rw [if_neg (by simp; omega)]
        simp
      · have result := mark_slice bytes p lower (by omega)
        have countEq : (bytes.length : Int)-p*p-1 = (n : Int)-1-p*p := by rw [size]; ring
        rw [countEq] at result
        simpa only [size, count] using result
  exact first.append (second.append (third.append fourth))

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

private def Invariant (n p : Nat) (bytes : List UInt8) : Prop :=
  Sieved (n-1) p bytes ∧ bytes[0]? = some 0 ∧ bytes[1]? = some 0

private theorem invariant_mark (n p : Nat) (bytes : List UInt8)
    (large : 3 ≤ n) (lower : 2 ≤ p) (invariant : Invariant n p bytes)
    (prime : Nat.Prime p) : Invariant n (p+1) (mark bytes p) := by
  refine ⟨sieved_mark (n-1) p bytes invariant.1 prime, ?_, ?_⟩
  · rw [mark_get bytes p 0 (by omega) (by have := invariant.1.1; omega)]
    simpa [show ¬ p^2 ≤ 0 by nlinarith] using invariant.2.1
  · rw [mark_get bytes p 1 (by omega) (by have := invariant.1.1; omega)]
    simpa [show ¬ p^2 ≤ 1 by nlinarith] using invariant.2.2

private def Stable (before after : HeapCalls.Env) : Prop :=
  after "n" = before "n" ∧ after "is_prime" = before "is_prime"

private theorem read_buffer (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat)
    (bytes : List UInt8) (valid : p < bytes.length)
    (getP : env "p" = some (number p)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    Evaluates program env (.subscript (.name "is_prime") (.name "p")) heap
      (number bytes[p].toNat) heap := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro h; cases h) (Evaluates.lookup getBuffer) (Evaluates.lookup getP)
  exact Mutable.getItem_buffer heap address p bytes found valid

private theorem iteration (n p : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (large : 3 ≤ n) (lower : 2 ≤ p)
    (inside : p*p < n) (getP : env "p" = some (number p))
    (getN : env "n" = some (number n)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (invariant : Invariant n p bytes) :
    ∃ finalEnv finalBytes, BlockContinues program env loopBody heap finalEnv
      (heap.set address (.bytearray finalBytes)) ∧ Invariant n (p+1) finalBytes ∧ Stable env finalEnv := by
  have size : bytes.length = n := by have := invariant.1.1; omega
  have valid : p < bytes.length := by rw [size]; nlinarith
  have live : bytes[p].toNat ≠ 0 ↔ Nat.Prime p :=
    (invariant.1.2 p bytes[p] lower (List.getElem?_eq_getElem valid)).trans
      (survives_current p lower)
  by_cases prime : Nat.Prime p
  · refine ⟨markEnv env n p, mark bytes p, ?_, invariant_mark n p bytes large lower invariant prime, ?_⟩
    · apply BlockContinues.branch (flag := true) (middle := heap)
        (read_buffer heap env address p bytes valid getP getBuffer found)
      · simp [Mutable.truth, scalarTruth, live.mpr prime]
      · exact mark_result env heap address n p bytes lower inside size getP getN getBuffer found
    · simp [Stable, markEnv, setLocal]
  · refine ⟨env, bytes, ?_, ⟨sieved_skip (n-1) p bytes invariant.1 prime, invariant.2⟩, rfl, rfl⟩
    rw [heap_set_same heap address _ found]
    apply BlockContinues.branch (flag := false) (middle := heap)
      (read_buffer heap env address p bytes valid getP getBuffer found)
    · have zero : bytes[p].toNat = 0 := by tauto
      simp [Mutable.truth, scalarTruth, zero]
    · exact .nil

private theorem loop (n first count : Nat) (bytes : List UInt8) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (address : Nat) (large : 3 ≤ n) (lower : 2 ≤ first)
    (endpoint : first + count = Nat.sqrt (n-1)+1)
    (getN : env "n" = some (number n)) (getBuffer : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) (invariant : Invariant n first bytes) :
    ∃ finalEnv finalBytes,
      BlockContinues program env [.rangeStepLoop "p" first (Nat.sqrt (n-1)+1) 1 loopBody []]
        heap finalEnv (heap.set address (.bytearray finalBytes)) ∧
      Invariant n (Nat.sqrt (n-1)+1) finalBytes ∧ Stable env finalEnv := by
  induction count generalizing first bytes env heap with
  | zero =>
      have same : first = Nat.sqrt (n-1)+1 := by omega
      refine ⟨env, bytes, ?_, by simpa only [same] using invariant, rfl, rfl⟩
      rw [heap_set_same heap address _ found]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "p" (number first)
      have inside : first*first < n := by
        have below : first ≤ Nat.sqrt (n-1) := by omega
        have squared := Nat.mul_self_le_mul_self below
        have upper := Nat.sqrt_le (n-1)
        omega
      obtain ⟨middleEnv, middleBytes, iterated, nextInvariant, stable⟩ :=
        iteration n first bytes bound heap address large lower inside rfl
          (by simpa [bound, setLocal] using getN) (by simpa [bound, setLocal] using getBuffer)
          found invariant
      let middleHeap := heap.set address (.bytearray middleBytes)
      have middleFound : middleHeap[address]? = some (.bytearray middleBytes) :=
        List.getElem?_set_self (heap_inside found)
      obtain ⟨finalEnv, finalBytes, executed, finalInvariant, saved⟩ :=
        ih (first+1) middleBytes middleEnv middleHeap (by omega) (by omega)
          (stable.1.trans (by simpa [bound, setLocal] using getN))
          (stable.2.trans (by simpa [bound, setLocal] using getBuffer)) middleFound nextInvariant
      refine ⟨finalEnv, finalBytes, ?_, finalInvariant, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          middleHeap, List.set_set] using iterated.append executed
      · exact ⟨saved.1.trans (stable.1.trans (by simp [bound, setLocal])),
          saved.2.trans (stable.2.trans (by simp [bound, setLocal]))⟩

private theorem initial_updates (n : Nat) (large : 3 ≤ n) :
    ((List.replicate n (1 : UInt8)).set 0 0).set 1 0 = initial (n-1) := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k+3 := ⟨n-3, by omega⟩
  simp [initial, List.replicate_succ]

private def initialHeap (heap : Mutable.Heap) (n : Nat) : Mutable.Heap :=
  heap ++ [.bytearray [1], .bytearray (initial (n-1))]

private def initializedEnv (env : HeapCalls.Env) (address n : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "is_prime" (.ref address)) "limit" (number (Nat.sqrt (n-1)))

private theorem setup (n : Nat) (large : 3 ≤ n) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) :
    BlockContinues program env ((code.drop 1).take 4) heap
      (initializedEnv env (heap.length+1) n) (initialHeap heap n) := by
  let e1 := setLocal env "is_prime" (.ref (heap.length+1))
  let allocated := heap ++ [.bytearray [1], .bytearray (List.replicate n 1)]
  let zeroed := heap ++ [.bytearray [1], .bytearray ((List.replicate n 1).set 0 0)]
  have first : BlockContinues program env [code[1]'(by decide)] heap e1 allocated := by
    apply BlockContinues.assign
    apply Evaluates.binary (av := .ref heap.length) (middle := heap ++ [.bytearray [1]])
      (bv := number n) (afterRight := heap ++ [.bytearray [1]])
    · exact Evaluates.builtin (.cons Evaluates.constant .nil) rfl
    · exact Evaluates.lookup getN
    · simp [allocated, Mutable.binary, Mutable.allocateBuffer, Except.map,
        show (BinOp.mult == .mult) = true from rfl]
  have second : BlockContinues program e1 [code[2]'(by decide)] allocated e1 zeroed := by
    apply BlockContinues.branch (flag := true) (middle := allocated)
      (value := .scalar (.bool true))
    · exact Evaluates.compare (Evaluates.lookup (by simpa [e1, setLocal] using getN))
        Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply BlockContinues.setItem (value := number 0) (by intros; simp) Evaluates.constant
        (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant .nil))
      have write := Mutable.setItem_buffer allocated (heap.length+1) 0 (List.replicate n 1) 0
        (by simp [allocated]) (by simp; omega)
      simpa [allocated, zeroed] using write
  have third : BlockContinues program e1 [code[3]'(by decide)] zeroed e1 (initialHeap heap n) := by
    apply BlockContinues.branch (flag := true) (middle := zeroed)
      (value := .scalar (.bool true))
    · exact Evaluates.compare (Evaluates.lookup (by simpa [e1, setLocal] using getN))
        Evaluates.constant (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply BlockContinues.setItem (value := number 0) (by intros; simp) Evaluates.constant
        (.cons (Evaluates.lookup rfl) (.cons Evaluates.constant .nil))
      have write := Mutable.setItem_buffer zeroed (heap.length+1) 1 ((List.replicate n 1).set 0 0) 0
        (by simp [zeroed]) (by simp; omega)
      simpa [zeroed, initialHeap, initial_updates n large] using write
  have fourth : BlockContinues program e1 [code[4]'(by decide)] (initialHeap heap n)
      (initializedEnv env (heap.length+1) n) (initialHeap heap n) := by
    apply BlockContinues.assign
    have argument : Evaluates program e1 (.binOp (.name "n") .sub (.constant (.int 1)))
        (initialHeap heap n) (number (n-1 : Nat)) (initialHeap heap n) := by
      exact Evaluates.binary (Evaluates.lookup (by simpa [e1, setLocal] using getN))
        Evaluates.constant (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub (by omega : 1 ≤ n)])
    apply Evaluates.library_call (.cons argument .nil) rfl
    change scalarResult (some (Math.isqrtCall [.int (n-1 : Nat)])) (initialHeap heap n) = _
    have root : Math.isqrt (n-1) = Nat.sqrt (n-1) := Nat.eq_sqrt.mpr (Math.isqrt_spec _)
    simp [scalarResult, Math.isqrtCall, root, show ¬ ((n-1 : Nat) : Int) < 0 by omega]
  exact first.append (second.append (third.append fourth))

/-- The exact returned bytearray covers every index below n, including the two nonprimes. -/
theorem result (n : Nat) (large : 3 ≤ n) (heap : Mutable.Heap) :
    ∃ bytes, ReturnsValue program "prime_sieve" [number n] heap (.ref (heap.length+1))
      (heap ++ [.bytearray [1], .bytearray bytes]) ∧ bytes.length = n ∧
      (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  let env : HeapCalls.Env := functionEnv
    ((findFunctionIn program.body "prime_sieve").getD ([], [])).1
    ((findFunctionIn program.body "prime_sieve").getD ([], [])).2 [number n] []
  let initialized := initializedEnv env (heap.length+1) n
  have initialInvariant : Invariant n 2 (initial (n-1)) :=
    ⟨sieved_initial (n-1) (by omega), by simp [initial], by simp [initial]⟩
  have rootPositive : 1 ≤ Nat.sqrt (n-1) := by
    have : 1 ≤ n-1 := by omega
    simpa using Nat.sqrt_le_sqrt this
  obtain ⟨finalEnv, bytes, executed, sieved, stable⟩ :=
    loop n 2 (Nat.sqrt (n-1)-1) (initial (n-1)) initialized (initialHeap heap n)
      (heap.length+1) large (by omega) (by omega)
      rfl
      (by simp [initialized, initializedEnv, setLocal]) (by simp [initialHeap]) initialInvariant
  have loopRun : BlockContinues program initialized [code[5]'(by decide)]
      (initialHeap heap n) finalEnv (heap ++ [.bytearray [1], .bytearray bytes]) := by
    apply BlockContinues.for_range (middle := initialHeap heap n)
      (.cons Evaluates.constant (.cons
        (Evaluates.binary (av := number (Nat.sqrt (n-1))) (bv := number 1)
          (value := number (Nat.sqrt (n-1)+1 : Nat)) (middle := initialHeap heap n) (afterRight := initialHeap heap n)
          (Evaluates.lookup rfl) Evaluates.constant
          (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
    simpa [initialHeap] using executed
  refine ⟨bytes, ?_, by have := sieved.1.1; omega, ?_⟩
  · apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := false) (middle := heap) (value := .scalar (.bool false))
    · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
    · rfl
    · apply BlockContinues.returns ((setup n large env heap rfl).append loopRun)
      apply BlockReturns.return_value
      exact Evaluates.lookup (stable.2.trans (by simp [initialized, initializedEnv, setLocal]))
  · intro i byte found
    by_cases lower : 2 ≤ i
    · have bound := (List.getElem?_eq_some_iff.mp found).1
      have finished : n-1 < (Nat.sqrt (n-1)+1)^2 := Nat.lt_succ_sqrt' _
      exact (sieved.1.2 i byte lower found).trans
        (survives_finished (n-1) (Nat.sqrt (n-1)+1) i lower (by have := sieved.1.1; omega) finished)
    · have cases : i = 0 ∨ i = 1 := by omega
      rcases cases with rfl | rfl
      · rw [sieved.2.1] at found
        cases found
        simp [Nat.not_prime_zero]
      · rw [sieved.2.2] at found
        cases found
        simp [Nat.not_prime_one]

end prime_sieve
namespace rotations_of
open ProjectEulerStatements.CircularPrimes

private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "rotations_of").getD ([], [])).2
private abbrev loopBody : List Stmt :=
  match (code[6]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []
private abbrev encode (values : List Nat) : List Mutable.Value := values.map (fun n : Nat => number n)
private def tail (width : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | count+1, x => let next := rotateValue width x; next :: tail width count next

private theorem step_value (width x : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getX : env "x" = some (number x))
    (getPower : env "pow10" = some (number (10^(width-1) : Nat))) :
    Evaluates program env
      (.binOp (.binOp (.binOp (.name "x") .mod (.name "pow10")) .mult (.constant (.int 10)))
        .add (.binOp (.name "x") .floorDiv (.name "pow10")))
      heap (number (rotateValue width x)) heap := by
  apply Evaluates.binary (av := number ((x % 10^(width-1))*10 : Nat))
    (bv := number (x / 10^(width-1) : Nat)) (middle := heap) (afterRight := heap)
  · apply Evaluates.binary (av := number (x % 10^(width-1) : Nat)) (bv := number 10)
      (middle := heap) (afterRight := heap)
    · exact Evaluates.binary (Evaluates.lookup getX) (Evaluates.lookup getPower)
        (by simp [Mutable.binary, scalarBinary, number ])
    · exact Evaluates.constant
    · simp [Mutable.binary, scalarBinary, number]
  · exact Evaluates.binary (Evaluates.lookup getX) (Evaluates.lookup getPower)
      (by simp [Mutable.binary, scalarBinary, number ])
  · simp [Mutable.binary, scalarBinary, number, rotateValue]

private theorem loop (width first count x : Nat) (endpoint : first+count = width-1)
    (before : List Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getX : env "x" = some (number x))
    (getPower : env "pow10" = some (number (10^(width-1) : Nat)))
    (getRots : env "rots" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "_" first (width-1) 1 loopBody []]
      (base ++ [.list (encode before)]) finalEnv
      (base ++ [.list (encode (before ++ tail width count x))]) ∧
      finalEnv "rots" = some (.ref base.length) := by
  induction count generalizing first x before env with
  | zero =>
      refine ⟨env, ?_, getRots⟩
      simpa [tail] using
        (BlockContinues.range_done (module := program) (env := env)
          (name := "_") (next := (first : Int)) (stop := (width : Int)-1) (stride := 1)
          (body := loopBody) (otherwise := [])
          (heap := base ++ [.list (encode before)])
          (by simp [Calls.rangeContinues]; omega) BlockContinues.nil)
  | succ count ih =>
      let bound := setLocal env "_" (number first)
      let updated := setLocal bound "x" (number (rotateValue width x))
      have step : BlockContinues program bound [loopBody[0]'(by decide)]
          (base ++ [.list (encode before)]) updated (base ++ [.list (encode before)]) :=
        BlockContinues.assign (step_value width x bound _
          (by simpa [bound, setLocal] using getX) (by simpa [bound, setLocal] using getPower))
      have appended : BlockContinues program updated [loopBody[1]'(by decide)]
          (base ++ [.list (encode before)]) updated
          (base ++ [.list (encode (before ++ [rotateValue width x]))]) := by
        apply BlockContinues.discard
        have evaluated := Evaluates.append (globals := []) (module := program) (env := updated)
          (base := .name "rots") (expression := .name "x")
          (value := number (rotateValue width x))
          (heap := base ++ [.list (encode before)])
          (Evaluates.lookup (by simpa [updated, bound, setLocal] using getRots))
          (by simp : (base ++ [.list (encode before)])[base.length]? = some (.list (encode before)))
          (Evaluates.lookup (by simp [updated, setLocal]))
          (by simp : (base ++ [.list (encode before)])[base.length]? = some (.list (encode before)))
        simpa [encode, List.set_append_right] using evaluated
      obtain ⟨finalEnv, rest, finalRots⟩ := ih (first+1) (rotateValue width x) (by omega)
        (before ++ [rotateValue width x]) updated rfl
        (by simpa [updated, bound, setLocal] using getPower)
        (by simpa [updated, bound, setLocal] using getRots)
      refine ⟨finalEnv, ?_, finalRots⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, tail, List.append_assoc]
        using (step.append appended).append rest

private theorem decimal_length (n : Nat) (positive : 0 < n) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (getS : env "s" = some (.scalar (.string (toString (n : Int))))) :
    Evaluates program env (.builtinCall .len [.name "s"]) heap
      (number (ProjectEulerStatements.P35.digitsBE n).length) heap := by
  apply Evaluates.builtin (.cons (Evaluates.lookup getS) .nil)
  have equal := ProjectEulerStatements.DecimalDigits.int_string_length n positive
  simpa [primitive, Mutable.length, number, ProjectEulerStatements.P35.digitsBE,
    List.length_reverse, equal]

private theorem result_raw (n : Nat) (positive : 0 < n) (heap : Mutable.Heap) :
    ReturnsValue program "rotations_of" [number n] heap (.ref heap.length)
      (heap ++ [.list (encode (n :: tail (ProjectEulerStatements.P35.digitsBE n).length
        ((ProjectEulerStatements.P35.digitsBE n).length-1) n))]) := by
  let width := (ProjectEulerStatements.P35.digitsBE n).length
  have widthPositive : 0 < width := by
    have nonempty := (Nat.digits_ne_nil_iff_ne_zero (b := 10)).mpr (Nat.ne_of_gt positive)
    simpa [width, ProjectEulerStatements.P35.digitsBE] using List.length_pos_iff.mpr nonempty
  let env : HeapCalls.Env := functionEnv
    ((findFunctionIn program.body "rotations_of").getD ([], [])).1
    ((findFunctionIn program.body "rotations_of").getD ([], [])).2 [number n] []
  let e1 := setLocal env "s" (.scalar (.string (toString (n : Int))))
  let e2 := setLocal e1 "k" (number width)
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (value := .scalar (.string (toString (n : Int)))) (middle := heap)
  · exact Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil) rfl
  · apply BlockReturns.assign (value := number width) (middle := heap)
    · exact decimal_length n positive e1 heap rfl
    · by_cases single : width = 1
      · apply BlockReturns.branch (flag := true) (middle := heap) (value := .scalar (.bool true))
        · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, single])
        · rfl
        · apply BlockReturns.return_value
          have listed : Evaluates program e2 (.list [.name "n"]) heap (.ref heap.length)
              (heap ++ [.list (encode [n])]) := Evaluates.list (.cons (Evaluates.lookup rfl) .nil)
          change Evaluates program e2 (.list [.name "n"]) heap (.ref heap.length)
            (heap ++ [.list (encode (n :: tail width (width-1) n))])
          rw [single]
          exact listed
      · apply BlockReturns.branch (flag := false) (middle := heap) (value := .scalar (.bool false))
        · exact Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)
        · rfl
        · let e3 := setLocal e2 "pow10" (number (10^(width-1) : Nat))
          let e4 := setLocal e3 "rots" (.ref heap.length)
          let e5 := setLocal e4 "x" (number n)
          apply BlockReturns.assign (value := number (10^(width-1) : Nat)) (middle := heap)
          · apply Evaluates.binary (av := number 10) (bv := number (width-1 : Nat))
              (middle := heap) (afterRight := heap) Evaluates.constant
            · exact Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant
                (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub widthPositive])
            · simp [Mutable.binary, scalarBinary, number, show ¬ ((width-1 : Nat) : Int) < 0 by omega]
          · apply BlockReturns.assign (value := .ref heap.length)
              (middle := heap ++ [.list (encode [n])])
            · exact Evaluates.list (.cons (Evaluates.lookup rfl) .nil)
            · apply BlockReturns.assign (value := number n) (middle := heap ++ [.list (encode [n])])
              · exact Evaluates.lookup rfl
              · obtain ⟨finalEnv, executed, finalRots⟩ := loop width 0 (width-1) n (by omega) [n] heap e5 rfl
                  (by simp [e5, e4, e3, setLocal]) (by simp [e5, e4, setLocal])
                have forLoop : BlockContinues program e5 [code[6]'(by decide)]
                    (heap ++ [.list (encode [n])]) finalEnv
                    (heap ++ [.list (encode (n :: tail width (width-1) n))]) := by
                  have upper : Evaluates program e5 (.binOp (.name "k") .sub (.constant (.int 1)))
                      (heap ++ [.list (encode [n])]) (number ((width : Int)-1))
                      (heap ++ [.list (encode [n])]) :=
                    Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl
                  apply BlockContinues.for_range (middle := heap ++ [.list (encode [n])])
                    (.cons upper .nil) rfl
                  simpa using executed
                exact BlockContinues.returns forLoop (BlockReturns.return_value (Evaluates.lookup finalRots))

private theorem tail_eq_map (width count x : Nat) :
    tail width count x = (List.range count).map (fun i => (rotateValue width)^[i+1] x) := by
  induction count generalizing x with
  | zero => simp [tail]
  | succ count ih =>
      rw [tail, ih, List.range_succ_eq_map]
      simp only [List.map_cons, List.map_map, Function.comp_def]
      congr 1


/-- Rotations retain their original decimal width, including a leading zero after rotation. -/
theorem result (n : Nat) (positive : 0 < n) (heap : Mutable.Heap) :
    ReturnsValue program "rotations_of" [number n] heap (.ref heap.length)
      (heap ++ [.list (encode (rotationValues n))]) := by
  have widthPositive : 0 < (ProjectEulerStatements.P35.digitsBE n).length := by
    have nonempty := (Nat.digits_ne_nil_iff_ne_zero (b := 10)).mpr (Nat.ne_of_gt positive)
    simpa [ProjectEulerStatements.P35.digitsBE] using List.length_pos_iff.mpr nonempty
  have same : n :: tail (ProjectEulerStatements.P35.digitsBE n).length
      ((ProjectEulerStatements.P35.digitsBE n).length-1) n = rotationValues n := by
    rw [← arithmetic_rotations n positive, tail_eq_map]
    obtain ⟨k, length⟩ : ∃ k, (ProjectEulerStatements.P35.digitsBE n).length = k+1 :=
      ⟨(ProjectEulerStatements.P35.digitsBE n).length-1, by omega⟩
    rw [length, Nat.add_sub_cancel, List.range_succ_eq_map]
    simp [List.map_map, Function.comp_def]
  simpa only [same] using result_raw n positive heap

end rotations_of

private abbrev encoded (values : List Nat) : List Mutable.Value := values.map (fun n : Nat => number n)

private theorem all_prime_loop (values remaining : List Nat) (position address sieveAddress : Nat)
    (bytes : List UInt8) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (encoded values)))
    (sieveFound : heap[sieveAddress]? = some (.bytearray bytes))
    (getSieve : env "is_prime" = some (.ref sieveAddress))
    (primes : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i))
    (bounds : ∀ n ∈ values, n < bytes.length) (suffix : values.drop position = remaining) :
    BlockReturns program env
      [.sequenceStepLoop "r" (.ref address) position
        (truthBody true (.subscript (.name "is_prime") (.name "r"))) [],
       .return (.constant (.bool true))] heap
      (.scalar (.bool (remaining.all (fun n => decide (Nat.Prime n))))) heap := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (encoded values), heap) := by
    simp [Mutable.sequence, found]
  induction remaining generalizing position env with
  | nil =>
      apply BlockReturns.sequence_done (source := .ref address) (items := encoded values) sequence
      · have missing : values[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
        simp [encoded, missing]
      · exact BlockReturns.return_value Evaluates.constant
  | cons head tail ih =>
      have item : values[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      have valid := bounds head (List.mem_of_getElem? item)
      have live := primes head bytes[head] (List.getElem?_eq_getElem valid)
      let bound := setLocal env "r" (number head)
      have evaluated : Evaluates program bound (.subscript (.name "is_prime") (.name "r"))
          heap (number bytes[head].toNat) heap := by
        apply Evaluates.subscript (middle := heap) (afterIndex := heap)
          (by intros; intro h; cases h)
          (Evaluates.lookup (by simpa [bound, setLocal] using getSieve)) (Evaluates.lookup rfl)
        exact Mutable.getItem_buffer heap sieveAddress head bytes sieveFound valid
      apply BlockReturns.sequence_step (source := .ref address) (items := encoded values) (value := number head) sequence (by simp [encoded, item])
      by_cases prime : Nat.Prime head
      · simp only [List.all_cons, prime, decide_true, Bool.true_and]
        apply BlockReturns.branch (flag := true) evaluated
          (by simp [Mutable.truth, scalarTruth, live.mpr prime])
        have rest := ih (position+1) bound
          (by simpa [bound, setLocal] using getSieve)
          (by simp [List.drop_add_one_eq_tail_drop, suffix])
        simpa [truthBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest
      · simp only [List.all_cons, prime, decide_false, Bool.false_and]
        apply BlockReturns.branch (flag := false) evaluated
          (by have zero : bytes[head].toNat = 0 := by tauto
              simp [Mutable.truth, scalarTruth, zero])
        exact BlockReturns.return_value Evaluates.constant

private theorem all_primes_evaluates (values : List Nat) (address sieveAddress : Nat)
    (bytes : List UInt8) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (encoded values)))
    (sieveFound : heap[sieveAddress]? = some (.bytearray bytes))
    (getRots : env "rots" = some (.ref address))
    (getSieve : env "is_prime" = some (.ref sieveAddress))
    (primes : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i))
    (bounds : ∀ n ∈ values, n < bytes.length) :
    Evaluates program env (.consumeGenerator (.builtinCall .all [])
      (.subscript (.name "is_prime") (.name "r")) [(.name "r", .name "rots", [])])
      heap (.scalar (.bool (values.all (fun n => decide (Nat.Prime n))))) heap := by
  apply Evaluates.truth_generator_sequence (universal := true) (items := encoded values)
    (by intro expressions h; cases h) (by intro expressions h; cases h)
    (by intro expressions h; cases h) (Evaluates.lookup getRots)
    (by simp [Mutable.sequence, found])
  apply all_prime_loop values values 0 address sieveAddress bytes heap _ found sieveFound
  · simpa [comprehensionTargetsEnv, targetLocals, comprehensionEnv, setLocal] using getSieve
  · exact primes
  · exact bounds
  · rfl

private theorem allowed_contains (character : Char) :
    Strings.contains "1379" (String.singleton character) =
      ProjectEulerStatements.CircularPrimes.allowedCharacter character := by
  rw [Strings.contains_eq_isInfix, String.toList_singleton]
  apply Bool.eq_iff_iff.mpr
  rw [List.isInfixOf_internal_iff_isInfix, List.singleton_infix_iff]
  simp only [ProjectEulerStatements.CircularPrimes.allowedCharacter, List.contains_iff_mem]
  rfl

private abbrev characterValue (character : Char) : Mutable.Value :=
  .scalar (.string (String.singleton character))

private theorem bad_character_loop (text : String) (remaining : List Char) (position : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getAllowed : env "allowed_digits" = some (.scalar (.string "1379")))
    (suffix : text.toList.drop position = remaining) :
    BlockReturns program env
      [.sequenceStepLoop "ch" (.scalar (.string text)) position
        (truthBody false (.membership (.name "ch") (.name "allowed_digits") true)) [],
       .return (.constant (.bool false))] heap
      (.scalar (.bool (remaining.any (fun ch => !ProjectEulerStatements.CircularPrimes.allowedCharacter ch)))) heap := by
  have sequence : Mutable.sequence (.scalar (.string text)) heap =
      some (.ok (text.toList.map characterValue), heap) := by rfl
  induction remaining generalizing position env with
  | nil =>
      apply BlockReturns.sequence_done (source := .scalar (.string text)) (items := text.toList.map characterValue) sequence
      · have missing : text.toList[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := text.toList) (i := position)).symm
        simp [missing]
      · exact BlockReturns.return_value Evaluates.constant
  | cons head tail ih =>
      have item : text.toList[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := text.toList) (i := position)).symm
      let bound := setLocal env "ch" (characterValue head)
      have evaluated : Evaluates program bound (.membership (.name "ch") (.name "allowed_digits") true)
          heap (.scalar (.bool (!ProjectEulerStatements.CircularPrimes.allowedCharacter head))) heap := by
        apply Evaluates.membership (.cons (Evaluates.lookup rfl)
          (.cons (Evaluates.lookup (by simpa [bound, setLocal] using getAllowed)) .nil))
        simpa only [allowed_contains] using HeapCalls.membership_string heap "1379" (String.singleton head) true
      apply BlockReturns.sequence_step (source := .scalar (.string text)) (items := text.toList.map characterValue) (value := characterValue head) sequence (by simp [item])
      cases allowed : ProjectEulerStatements.CircularPrimes.allowedCharacter head with
      | false =>
          simp only [List.any_cons, allowed, Bool.not_false, Bool.true_or]
          apply BlockReturns.branch (flag := true) (by simpa [allowed] using evaluated) rfl
          exact BlockReturns.return_value Evaluates.constant
      | true =>
          simp only [List.any_cons, allowed, Bool.not_true, Bool.false_or]
          apply BlockReturns.branch (flag := false) (by simpa [allowed] using evaluated) rfl
          have rest := ih (position+1) bound
            (by simpa [bound, setLocal] using getAllowed)
            (by simp [List.drop_add_one_eq_tail_drop, suffix])
          simpa [truthBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using rest

private theorem bad_characters_evaluates (text : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getText : env "s" = some (.scalar (.string text)))
    (getAllowed : env "allowed_digits" = some (.scalar (.string "1379"))) :
    Evaluates program env (.consumeGenerator (.builtinCall .any [])
      (.membership (.name "ch") (.name "allowed_digits") true) [(.name "ch", .name "s", [])])
      heap (.scalar (.bool (text.toList.any
        (fun ch => !ProjectEulerStatements.CircularPrimes.allowedCharacter ch)))) heap := by
  apply Evaluates.truth_generator_sequence (universal := false)
    (by intro expressions h; cases h) (by intro expressions h; cases h)
    (by intro expressions h; cases h) (Evaluates.lookup getText)
    (by rfl : Mutable.sequence (.scalar (.string text)) heap =
      some (.ok (text.toList.map characterValue), heap))
  apply bad_character_loop text text.toList 0 heap _
  · simpa [comprehensionTargetsEnv, targetLocals] using getAllowed
  · rfl

private abbrev scalarNaturals (items : List Nat) : List Constant := items.map (fun n : Nat => .int n)
private abbrev countFilter : Expr := .compare (.name "r") [(.lt, .name "limit")]
private abbrev countBody := comprehensionFilters [countFilter] (sumBody (.constant (.int 1)) [])
private def belowWeight (limit : Nat) : Constant → Int
  | .int n => if n < limit then 1 else 0
  | _ => 0

private theorem belowWeight_sum (limit : Nat) (items : List Nat) :
    ((scalarNaturals items).map (belowWeight limit)).sum =
      ((items.filter (fun n => decide (n < limit))).length : Nat) := by
  induction items with
  | nil => rfl
  | cons head tail ih =>
      change belowWeight limit (.int head) + ((scalarNaturals tail).map (belowWeight limit)).sum = _
      rw [ih]
      by_cases inside : head < limit <;>
        simp [belowWeight, inside, Nat.cast_add, add_comm]

private theorem count_below_evaluates (limit address : Nat) (items : List Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.set (scalarNaturals items)))
    (getSet : env "uniq" = some (.ref address)) (getLimit : env "limit" = some (number limit)) :
    SetCalls.Evaluates program env (.sumGenerator (.constant (.int 1))
      [("r", .name "uniq", [countFilter])] (.constant (.int 0))) heap
      (number (items.filter (fun n => decide (n < limit))).length) heap := by
  apply SetCalls.Evaluates.sum_generator_set (by intros; simp)
    (Evaluates.lookup getSet) found Evaluates.constant rfl
  intro choices anchors stack used
  let generator := setLocal (comprehensionEnv env [("r", .name "uniq", [countFilter])])
    sumAccumulator (number 0)
  have iterate (inner : HeapCalls.Env) (total : Int) (item : Constant)
      (valid : inner "limit" = some (number limit)) (member : item ∈ scalarNaturals items)
      (accumulator : inner sumAccumulator = some (number total)) :
      ∃ updated, BlockContinues program (setLocal inner "r" (.scalar item)) countBody heap updated heap ∧
        updated "limit" = some (number limit) ∧
        updated sumAccumulator = some (number (total + belowWeight limit item)) := by
    obtain ⟨n, _, rfl⟩ := List.mem_map.mp member
    let bound := setLocal inner "r" (number n)
    have tested : Evaluates program bound countFilter heap (.scalar (.bool (decide (n < limit)))) heap :=
      Evaluates.compare (Evaluates.lookup rfl)
        (Evaluates.lookup (by simpa [bound, setLocal] using valid))
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
    by_cases inside : n < limit
    · let updated := setLocal bound sumAccumulator (number (total+1))
      refine ⟨updated, ?_, by simpa [updated, bound, setLocal, sumAccumulator] using valid, ?_⟩
      · apply BlockContinues.branch (flag := true) (by simpa [inside] using tested) rfl
        apply BlockContinues.assign (value := number (total+1)) (finalHeap := heap)
        exact Evaluates.binary
          (Evaluates.lookup (by simpa [bound, setLocal, sumAccumulator] using accumulator))
          Evaluates.constant rfl
      · simp [updated, setLocal, belowWeight, inside]
    · refine ⟨bound, ?_, by simpa [bound, setLocal] using valid, ?_⟩
      · exact BlockContinues.branch (flag := false) (by simpa [inside] using tested) rfl .nil
      · simpa [bound, setLocal, belowWeight, inside, sumAccumulator] using accumulator
  have result := SetCalls.sum_map_loop choices program heap address (scalarNaturals items) (scalarNaturals items)
    found (fun _ member => member) "r" countBody (belowWeight limit)
    (fun inner => inner "limit" = some (number limit)) rfl iterate 0 generator
    (by simpa [generator, setLocal, sumAccumulator, comprehensionEnv] using getLimit)
    (by simp [generator, setLocal]) anchors stack used
  simpa only [Int.zero_add, belowWeight_sum] using result

open ProjectEulerStatements.CircularPrimes

private theorem set_rotations_evaluates (n address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getRots : env "rots" = some (.ref address))
    (found : heap[address]? = some (.list ((rotationValues n).map (fun r : Nat => number r)))) :
    Evaluates program env (.builtinCall .set [.name "rots"]) heap (.ref heap.length)
      (heap ++ [.set (scalarNaturals (rotationValues n).eraseDups)]) := by
  apply Evaluates.builtin (.cons (Evaluates.lookup getRots) .nil)
  simpa only [primitive, scalarNaturals, ScalarSet.fold_insert_nats] using
    Mutable.setFrom_list heap address (scalarNaturals (rotationValues n))
      (by simpa [scalarNaturals, number, List.map_map] using found)

/- The only schedule-dependent operation is the final generator sum. Every
other statement can use the ordinary shared-heap rules before this boundary. -/
private def SetIteration (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap) : Prop :=
  ∀ next following choices anchors stack used, ∃ finalUsed,
    SetCalls.Reaches choices program anchors
      ⟨⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩, used⟩
      ⟨⟨.block (next :: following) finalEnv, stack, finalHeap⟩, finalUsed⟩

private theorem SetIteration.done (env : HeapCalls.Env) (heap : Mutable.Heap) :
    SetIteration env [] heap env heap := by
  intro next following choices anchors stack used
  exact ⟨used, .next rfl .refl⟩

private theorem SetIteration.prepend
    (first : BlockContinues program env body heap middleEnv middleHeap)
    (rest : SetIteration middleEnv tail middleHeap finalEnv finalHeap) :
    SetIteration env (body ++ tail) heap finalEnv finalHeap := by
  intro next following choices anchors stack used
  obtain ⟨finalUsed, path⟩ := rest next following choices anchors stack used
  refine ⟨finalUsed, ?_⟩
  have initialPath := SetCalls.Reaches.of_heap
    (first (tail ++ (.loopResume next :: following)) anchors stack) choices used
  have tailRun : SetCalls.Reaches choices program anchors
      ⟨⟨.block (tail ++ (.loopResume next :: following)) middleEnv, stack, middleHeap⟩, used⟩
      ⟨⟨.block (next :: following) finalEnv, stack, finalHeap⟩, finalUsed⟩ := by
    simpa only [List.append_assoc, List.singleton_append] using path
  simpa only [List.append_assoc, List.singleton_append] using initialPath.trans tailRun

private theorem SetIteration.branch
    (evaluated : Evaluates program env condition heap value middle)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : SetIteration env ((if flag then yes else no) ++ body) middle finalEnv finalHeap) :
    SetIteration env (.ifThen condition yes no :: body) heap finalEnv finalHeap := by
  intro next following choices anchors stack used
  obtain ⟨finalUsed, path⟩ := rest next following choices anchors stack used
  refine ⟨finalUsed, .next rfl ?_⟩
  apply (SetCalls.Reaches.of_heap (evaluated anchors _) choices used).trans
  apply SetCalls.Reaches.next (following :=
    ⟨⟨.block (((if flag then yes else no) ++ body) ++ [.loopResume next] ++ following) env,
      stack, middle⟩, used⟩)
  · simp [SetCalls.step, HeapCalls.step, run, truth, SetCalls.liftStep, List.append_assoc]
  · exact path

private theorem SetIteration.skip
    (target : ∀ next following,
      Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following)) :
    SetIteration env (.continueLoop :: body) heap env heap := by
  intro next following choices anchors stack used
  have found : Calls.loopTarget (body ++ (.loopResume next :: following)) = some (next, following) := by
    simpa only [List.append_assoc, List.singleton_append] using target next following
  exact ⟨used, .next (by simp [SetCalls.step, HeapCalls.step, found, SetCalls.liftStep]) .refl⟩

private theorem SetIteration.augAssign
    (lookup : env name = some left)
    (evaluated : SetCalls.Evaluates program env expression heap right middle)
    (operation : inplace op left right middle = some (.ok value, finalHeap)) :
    SetIteration env [.augAssign (.name name) op expression] heap
      (setLocal env name value) finalHeap := by
  intro next following choices anchors stack used
  obtain ⟨finalUsed, path⟩ := evaluated choices anchors
    (.binaryRight op left true :: .assign name (.loopResume next :: following) env :: stack) used
  refine ⟨finalUsed, .next rfl ?_⟩
  apply (SetCalls.Reaches.of_heap (Evaluates.lookup (module := program) (heap := heap)
    lookup anchors _) choices used).trans
  apply SetCalls.Reaches.next rfl
  apply path.trans
  apply SetCalls.Reaches.next (following :=
    ⟨⟨.value value, .assign name (.loopResume next :: following) env :: stack, finalHeap⟩, finalUsed⟩)
  · simp [SetCalls.step, HeapCalls.step, result, run, operation, SetCalls.liftStep]
  · exact .next rfl (.next rfl .refl)

private def processedBytes (limit : Nat) (processed : Finset Nat) : List UInt8 :=
  (List.range limit).map (fun i => if i ∈ processed then 1 else 0)

private theorem processedBytes_length (limit : Nat) (processed : Finset Nat) :
    (processedBytes limit processed).length = limit := by simp [processedBytes]

private theorem processedBytes_get (limit : Nat) (processed : Finset Nat) (i : Nat) (inside : i < limit) :
    (processedBytes limit processed)[i]? = some (if i ∈ processed then 1 else 0) := by
  simp [processedBytes, List.getElem?_range inside]

private theorem processedBytes_set (limit : Nat) (processed : Finset Nat) (p : Nat) (inside : p < limit) :
    (processedBytes limit processed).set p 1 = processedBytes limit (insert p processed) := by
  apply List.ext_getElem?
  intro i
  by_cases valid : i < limit
  · by_cases same : i = p
    · subst i
      rw [List.getElem?_set_self (by simpa [processedBytes_length] using inside),
        processedBytes_get _ _ _ inside]
      simp
    · rw [List.getElem?_set_ne (Ne.symm same), processedBytes_get _ _ _ valid,
        processedBytes_get _ _ _ valid]
      simp [same]
  · rw [List.getElem?_eq_none (by simp [processedBytes_length]; omega),
      List.getElem?_eq_none (by simp [processedBytes_length]; omega)]

private theorem write_processed (limit p address : Nat) (processed : Finset Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (name : String) (inside : p < limit)
    (getBuffer : env "processed" = some (.ref address)) (getP : env name = some (number p))
    (found : heap[address]? = some (.bytearray (processedBytes limit processed))) :
    BlockContinues program env
      [.assign [.subscript (.name "processed") (.name name)] (.constant (.int 1))]
      heap env (heap.set address (.bytearray (processedBytes limit (insert p processed)))) := by
  apply BlockContinues.setItem (value := number 1) (by intros; simp) Evaluates.constant
    (.cons (Evaluates.lookup getBuffer) (.cons (Evaluates.lookup getP) .nil))
  have write := Mutable.setItem_buffer heap address p (processedBytes limit processed) 1 found
    (by simpa [processedBytes_length] using inside)
  simpa [processedBytes_set limit processed p inside] using write

private def markRotations (limit : Nat) (processed : Finset Nat) (values : List Nat) : Finset Nat :=
  processed ∪ (values.filter (fun n => decide (n < limit))).toFinset

private theorem markRotations_cons (limit head : Nat) (processed : Finset Nat) (tail : List Nat) :
    markRotations limit processed (head :: tail) =
      markRotations limit (if head < limit then insert head processed else processed) tail := by
  by_cases inside : head < limit
  · ext n
    simp [markRotations, inside]
  · simp [markRotations, inside]

private abbrev markRotationBody : List Stmt :=
  [.ifThen (.compare (.name "r") [(.lt, .name "limit")])
    [.assign [.subscript (.name "processed") (.name "r")] (.constant (.int 1))] []]

private theorem mark_rotation_loop (limit position address listAddress : Nat)
    (values remaining : List Nat) (processed : Finset Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.bytearray (processedBytes limit processed)))
    (rotationsFound : heap[listAddress]? = some (.list (encoded values)))
    (getBuffer : env "processed" = some (.ref address)) (getLimit : env "limit" = some (number limit))
    (suffix : values.drop position = remaining) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "r" (.ref listAddress) position markRotationBody []] heap finalEnv
      (heap.set address (.bytearray (processedBytes limit (markRotations limit processed remaining)))) ∧
      (∀ name, name ≠ "r" → finalEnv name = env name) := by
  have separate : listAddress ≠ address := by
    intro same
    subst listAddress
    rw [found] at rotationsFound
    cases rotationsFound
  induction remaining generalizing position processed heap env with
  | nil =>
      have sequence : Mutable.sequence (.ref listAddress) heap = some (.ok (encoded values), heap) := by
        simp [Mutable.sequence, rotationsFound]
      refine ⟨env, ?_, fun _ _ => rfl⟩
      have equal : heap.set address (.bytearray (processedBytes limit processed)) = heap :=
        prime_sieve.heap_set_same heap address _ found
      simp only [markRotations, List.filter_nil, List.toFinset_nil, Finset.union_empty, equal]
      apply BlockContinues.sequence_done (source := .ref listAddress) (items := encoded values) sequence
      · have missing : values[position]? = none := by
          simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
        simp [encoded, missing]
      · exact .nil
  | cons head tail ih =>
      have sequence : Mutable.sequence (.ref listAddress) heap = some (.ok (encoded values), heap) := by
        simp [Mutable.sequence, rotationsFound]
      have item : values[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      let bound := setLocal env "r" (number head)
      let nextProcessed := if head < limit then insert head processed else processed
      let nextHeap := heap.set address (.bytearray (processedBytes limit nextProcessed))
      have addressInside : address < heap.length := (List.getElem?_eq_some_iff.mp found).1
      have nextFound : nextHeap[address]? = some (.bytearray (processedBytes limit nextProcessed)) :=
        List.getElem?_set_self addressInside
      have nextRotations : nextHeap[listAddress]? = some (.list (encoded values)) := by
        simpa [nextHeap, List.getElem?_set_ne (Ne.symm separate)] using rotationsFound
      have tested : Evaluates program bound (.compare (.name "r") [(.lt, .name "limit")]) heap
          (.scalar (.bool (decide (head < limit)))) heap := by
        exact Evaluates.compare (Evaluates.lookup rfl)
          (Evaluates.lookup (by simpa [bound, setLocal] using getLimit))
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
      have marked : BlockContinues program bound markRotationBody heap bound nextHeap := by
        by_cases inside : head < limit
        · apply BlockContinues.branch (flag := true) (by simpa [inside] using tested) rfl
          simpa [nextHeap, nextProcessed, inside] using write_processed limit head address processed heap bound "r" inside
            (by simpa [bound, setLocal] using getBuffer) rfl found
        · have same : nextHeap = heap := by
            simp only [nextHeap, nextProcessed, if_neg inside]
            exact prime_sieve.heap_set_same heap address _ found
          rw [same]
          exact BlockContinues.branch (flag := false) (by simpa [inside] using tested) rfl .nil
      obtain ⟨finalEnv, rest, preserved⟩ := ih (position+1) nextProcessed nextHeap bound nextFound nextRotations
        (by simpa [bound, setLocal] using getBuffer) (by simpa [bound, setLocal] using getLimit)
        (by simp [List.drop_add_one_eq_tail_drop, suffix])
      refine ⟨finalEnv, ?_, ?_⟩
      · apply BlockContinues.sequence_step (source := .ref listAddress) (items := encoded values) (value := number head) sequence (by simp [encoded, item])
        simpa [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          nextHeap, List.set_set, markRotations_cons, nextProcessed] using marked.append rest
      · intro name other
        rw [preserved name other]
        simp [bound, setLocal, other]

namespace solve
open ProjectEulerStatements.CircularPrimes
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private abbrev loopBody : List Stmt :=
  match (code[7]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private def sieveSize (limit : Nat) : Nat := 10^(ProjectEulerStatements.P35.digitsBE (limit-1)).length

private theorem sieveSize_bound (limit : Nat) : limit ≤ sieveSize limit := by
  have bound := Nat.lt_base_pow_length_digits (m := limit-1) (by decide : 1 < 10)
  simpa [sieveSize, ProjectEulerStatements.P35.digitsBE] using (show limit ≤ 10^(Nat.digits 10 (limit-1)).length by omega)

private theorem sieveSize_large (limit : Nat) (large : 3 ≤ limit) : 3 ≤ sieveSize limit :=
  large.trans (sieveSize_bound limit)

private def initializedEnv (env : HeapCalls.Env) (base limit : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal (setLocal (setLocal env
    "d" (number (ProjectEulerStatements.P35.digitsBE (limit-1)).length))
    "sieve_n" (number (sieveSize limit))) "is_prime" (.ref (base+1)))
    "processed" (.ref (base+2))) "allowed_digits" (.scalar (.string "1379"))) "count" (number 0)

private theorem processedBytes_empty (limit : Nat) :
    processedBytes limit ∅ = List.replicate limit 0 := by
  simp [processedBytes, List.map_const', List.length_range]

private theorem setup (limit : Nat) (large : 3 ≤ limit) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) :
    ∃ bytes, BlockContinues program env ((code.drop 1).take 6) heap
      (initializedEnv env heap.length limit)
      (heap ++ [.bytearray [1], .bytearray bytes, .bytearray (processedBytes limit ∅)]) ∧
      bytes.length = sieveSize limit ∧
      (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) := by
  let width := (ProjectEulerStatements.P35.digitsBE (limit-1)).length
  let e1 := setLocal env "d" (number width)
  let e2 := setLocal e1 "sieve_n" (number (sieveSize limit))
  let e3 := setLocal e2 "is_prime" (.ref (heap.length+1))
  let e4 := setLocal e3 "processed" (.ref (heap.length+2))
  let e5 := setLocal e4 "allowed_digits" (.scalar (.string "1379"))
  have first : BlockContinues program env [code[1]'(by decide)] heap e1 heap := by
    apply BlockContinues.assign
    have subtracted : Evaluates program env (.binOp (.name "limit") .sub (.constant (.int 1)))
        heap (number (limit-1 : Nat)) heap :=
      Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary, number, Int.natCast_sub (by omega : 1 ≤ limit)])
    have text : Evaluates program env (.builtinCall .str [.binOp (.name "limit") .sub (.constant (.int 1))])
        heap (.scalar (.string (toString ((limit-1 : Nat) : Int)))) heap :=
      Evaluates.builtin (.cons subtracted .nil) rfl
    apply Evaluates.builtin (.cons text .nil)
    have length := ProjectEulerStatements.DecimalDigits.int_string_length (limit-1) (by omega)
    simpa [primitive, Mutable.length, number, width, ProjectEulerStatements.P35.digitsBE,
      List.length_reverse, length]
  have second : BlockContinues program e1 [code[2]'(by decide)] heap e2 heap := by
    apply BlockContinues.assign
    apply Evaluates.builtin (.cons
      (Evaluates.lookup (by simpa [e1, setLocal] using getLimit)) (.cons
      (Evaluates.binary (av := number 10) (bv := number width)
        (value := number (sieveSize limit)) (finalHeap := heap) (middle := heap) (afterRight := heap)
        Evaluates.constant (Evaluates.lookup rfl)
        (by simp [Mutable.binary, scalarBinary, number, sieveSize, width,
          show ¬ (width : Int) < 0 by omega])) .nil))
    by_cases greater : limit < sieveSize limit
    · simp [primitive, Calls.positionalMaximum, Calls.maximumFrom,
        scalarResult, scalarCompare, evalCmpOp, number, greater]
    · have equal : sieveSize limit = limit := by have := sieveSize_bound limit; omega
      simp [primitive, Calls.positionalMaximum, Calls.maximumFrom,
        scalarResult, scalarCompare, evalCmpOp, number, equal]
  obtain ⟨bytes, returned, length, primes⟩ := prime_sieve.result (sieveSize limit)
    (sieveSize_large limit large) heap
  let work := heap ++ [.bytearray [1], .bytearray bytes]
  let finalHeap := heap ++ [.bytearray [1], .bytearray bytes, .bytearray (processedBytes limit ∅)]
  have third : BlockContinues program e2 [code[3]'(by decide)] heap e3 work :=
    BlockContinues.assign (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) returned)
  have fourth : BlockContinues program e3 [code[4]'(by decide)] work e4 finalHeap := by
    apply BlockContinues.assign
    apply Evaluates.builtin (.cons
      (Evaluates.lookup (by simpa [e3, e2, e1, setLocal] using getLimit)) .nil)
    simp [primitive, Mutable.bytearray, Mutable.allocateBuffer,
      work, finalHeap, processedBytes_empty, Except.map, show ¬ (limit : Int) < 0 by omega]
  have fifth : BlockContinues program e4 [code[5]'(by decide)] finalHeap e5 finalHeap :=
    BlockContinues.assign Evaluates.constant
  have sixth : BlockContinues program e5 [code[6]'(by decide)] finalHeap
      (initializedEnv env heap.length limit) finalHeap := BlockContinues.assign Evaluates.constant
  exact ⟨bytes, first.append (second.append (third.append (fourth.append (fifth.append sixth)))), length, primes⟩

private structure ScanLocals (env : HeapCalls.Env) (limit sieveAddress processedAddress total : Nat) : Prop where
  limitValue : env "limit" = some (number limit)
  sieve : env "is_prime" = some (.ref sieveAddress)
  processed : env "processed" = some (.ref processedAddress)
  allowed : env "allowed_digits" = some (.scalar (.string "1379"))
  count : env "count" = some (number total)

private theorem ScanLocals.other (locals : ScanLocals env limit sieveAddress processedAddress total)
    (name : String) (value : Mutable.Value)
    (other : name ∉ ["limit", "is_prime", "processed", "allowed_digits", "count"]) :
    ScanLocals (setLocal env name value) limit sieveAddress processedAddress total := by
  rcases locals with ⟨limitValue, sieve, processed, allowed, count⟩
  simp only [List.mem_cons, not_or] at other
  constructor <;> simp_all [setLocal, eq_comm]

private theorem ScanLocals.updated (locals : ScanLocals env limit sieveAddress processedAddress total)
    (nextTotal : Nat) :
    ScanLocals (setLocal env "count" (number nextTotal)) limit sieveAddress processedAddress nextTotal := by
  rcases locals with ⟨limitValue, sieve, processed, allowed, count⟩
  constructor <;> simp_all [setLocal]

private abbrev skipCondition : Expr :=
  .boolOp .or [.unaryOp .not (.subscript (.name "is_prime") (.name "p")),
    .subscript (.name "processed") (.name "p")]

private theorem skip_evaluates (p limit sieveAddress processedAddress total : Nat)
    (processed : Finset Nat) (bytes : List UInt8) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (locals : ScanLocals env limit sieveAddress processedAddress total)
    (getP : env "p" = some (number p)) (inside : p < limit) (sieveInside : p < bytes.length)
    (sieveFound : heap[sieveAddress]? = some (.bytearray bytes))
    (processedFound : heap[processedAddress]? = some (.bytearray (processedBytes limit processed)))
    (primes : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) :
    ∃ value, Evaluates program env skipCondition heap value heap ∧
      Mutable.truth value heap = some (.ok (decide (¬Nat.Prime p ∨ p ∈ processed)), heap) := by
  have live := primes p bytes[p] (List.getElem?_eq_getElem sieveInside)
  have readPrime : Evaluates program env (.subscript (.name "is_prime") (.name "p")) heap
      (number bytes[p].toNat) heap := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (by intros; intro h; cases h)
      (Evaluates.lookup locals.sieve) (Evaluates.lookup getP)
    exact Mutable.getItem_buffer heap sieveAddress p bytes sieveFound sieveInside
  by_cases prime : Nat.Prime p
  · have tested : Evaluates program env (.unaryOp .not (.subscript (.name "is_prime") (.name "p")))
        heap (.scalar (.bool false)) heap :=
      Evaluates.unary readPrime (by simp [HeapCalls.unary, Mutable.truth, scalarTruth, live.mpr prime])
    let byte : UInt8 := if p ∈ processed then 1 else 0
    have readProcessed : Evaluates program env (.subscript (.name "processed") (.name "p"))
        heap (number byte.toNat) heap := by
      apply Evaluates.subscript (middle := heap) (afterIndex := heap) (by intros; intro h; cases h)
        (Evaluates.lookup locals.processed) (Evaluates.lookup getP)
      have valid : p < (processedBytes limit processed).length := by simpa [processedBytes_length] using inside
      have equal : (processedBytes limit processed)[p] = byte := by
        have found := processedBytes_get limit processed p inside
        rw [List.getElem?_eq_getElem valid, Option.some.injEq] at found
        exact found
      simpa only [equal] using Mutable.getItem_buffer heap processedAddress p
        (processedBytes limit processed) processedFound valid
    have truth : Mutable.truth (number byte.toNat) heap = some (.ok (decide (p ∈ processed)), heap) := by
      by_cases seen : p ∈ processed <;> simp [byte, seen, Mutable.truth, scalarTruth]
    refine ⟨number byte.toNat, ?_, by simpa [prime] using truth⟩
    exact Evaluates.boolean_next tested rfl rfl (Evaluates.boolean_last readProcessed truth)
  · have zero : bytes[p].toNat = 0 := by tauto
    have tested : Evaluates program env (.unaryOp .not (.subscript (.name "is_prime") (.name "p")))
        heap (.scalar (.bool true)) heap :=
      Evaluates.unary readPrime (by simp [HeapCalls.unary, Mutable.truth, scalarTruth, zero])
    exact ⟨.scalar (.bool true), Evaluates.boolean_stop tested rfl rfl, by simp [prime, Mutable.truth, scalarTruth]⟩

private theorem found_append {heap : Mutable.Heap} {address : Nat}
    (found : heap[address]? = some (object : Mutable.Object)) (extra : Mutable.Heap) :
    (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem ScanLocals.preserved
    (locals : ScanLocals before limit sieveAddress processedAddress total)
    (same : ∀ name, name ≠ "r" → after name = before name) :
    ScanLocals after limit sieveAddress processedAddress total := by
  constructor
  · rw [same "limit" (by decide)]; exact locals.limitValue
  · rw [same "is_prime" (by decide)]; exact locals.sieve
  · rw [same "processed" (by decide)]; exact locals.processed
  · rw [same "allowed_digits" (by decide)]; exact locals.allowed
  · rw [same "count" (by decide)]; exact locals.count

private theorem markRotations_cycle (limit p : Nat) (processed : Finset Nat)
    (positive : 0 < p) (inside : p < limit) :
    markRotations limit (insert p processed) (rotationValues p) = processed ∪ boundedOrbit limit p := by
  have cycle : ((rotationValues p).filter (fun n => decide (n < limit))).toFinset = boundedOrbit limit p := by
    ext n
    simp [boundedOrbit, orbit]
  have self : p ∈ boundedOrbit limit p := Finset.mem_filter.mpr ⟨mem_orbit_self p positive, inside⟩
  simp only [markRotations, cycle]
  ext n
  simp only [Finset.mem_union, Finset.mem_insert]
  constructor
  · rintro ((rfl | old) | member)
    · exact .inr self
    · exact .inl old
    · exact .inr member
  · intro member
    rcases member with old | member
    · exact .inl (.inr old)
    · exact .inr member

private theorem rotation_primes_eq (p : Nat) (lower : 2 ≤ p) :
    (rotationValues p).all (fun n => decide (Nat.Prime n)) = ProjectEulerStatements.P35.isCircularPrime p := by
  simp [ProjectEulerStatements.P35.isCircularPrime, show ¬ p < 2 by omega,
    listAll_map_eq_all, rotationValues, List.all_map, ProjectEulerStatements.P35.isPrime, Function.comp_def]

private theorem bad_digits_eq (p : Nat) (positive : 0 < p) :
    (toString (p : Int)).toList.any (fun ch => !allowedCharacter ch) =
      decide (¬AllowedDigits (ProjectEulerStatements.P35.digitsBE p)) := by
  have allowed : (toString (p : Int)).toList.all allowedCharacter =
      decide (AllowedDigits (ProjectEulerStatements.P35.digitsBE p)) := by
    apply Bool.eq_iff_iff.mpr
    simpa only [decide_eq_true_eq] using allowed_string_iff p positive
  rw [← List.not_all_eq_any_not, allowed]
  simp

private theorem iteration (limit sieveAddress processedAddress : Nat) (bytes : List UInt8)
    (p : Nat) (state : CountState) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (lower : 2 ≤ p) (inside : p < limit) (separate : sieveAddress ≠ processedAddress)
    (length : bytes.length = sieveSize limit)
    (primes : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i))
    (sieveFound : heap[sieveAddress]? = some (.bytearray bytes))
    (processedFound : heap[processedAddress]? = some (.bytearray (processedBytes limit state.1)))
    (locals : ScanLocals env limit sieveAddress processedAddress state.2)
    (getP : env "p" = some (number p)) :
    ∃ finalHeap finalEnv, SetIteration env loopBody heap finalEnv finalHeap ∧
      finalHeap[sieveAddress]? = some (.bytearray bytes) ∧
      finalHeap[processedAddress]? = some (.bytearray (processedBytes limit (algorithmStep limit state p).1)) ∧
      ScanLocals finalEnv limit sieveAddress processedAddress (algorithmStep limit state p).2 := by
  obtain ⟨skipValue, skipTested, skipTruth⟩ := skip_evaluates p limit sieveAddress processedAddress state.2
    state.1 bytes env heap locals getP inside (by rw [length]; exact inside.trans_le (sieveSize_bound limit))
    sieveFound processedFound primes
  by_cases skip : ¬ Nat.Prime p ∨ p ∈ state.1
  · refine ⟨heap, env, ?_, sieveFound, ?_, ?_⟩
    · apply SetIteration.branch (flag := true) skipTested (by simpa only [skip, decide_true] using skipTruth)
      apply SetIteration.skip
      intro next following
      rfl
    · simpa only [algorithmStep, if_pos skip] using processedFound
    · simpa only [algorithmStep, if_pos skip] using locals
  · let marked := heap.set processedAddress (.bytearray (processedBytes limit (insert p state.1)))
    have processedInside : processedAddress < heap.length := (List.getElem?_eq_some_iff.mp processedFound).1
    have markFound : marked[processedAddress]? = some (.bytearray (processedBytes limit (insert p state.1))) :=
      List.getElem?_set_self processedInside
    have markSieve : marked[sieveAddress]? = some (.bytearray bytes) := by
      simpa [marked, List.getElem?_set_ne separate.symm] using sieveFound
    have markedRun : BlockContinues program env [loopBody[1]'(by decide)] heap env marked :=
      write_processed limit p processedAddress state.1 heap env "p" inside locals.processed getP processedFound
    have smallTest : Evaluates program env (.compare (.name "p") [(.lt, .constant (.int 10))]) marked
        (.scalar (.bool (decide (p < 10)))) marked :=
      Evaluates.compare (Evaluates.lookup getP) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
    by_cases small : p < 10
    · let nextEnv := setLocal env "count" (number (state.2 + 1 : Nat))
      refine ⟨marked, nextEnv, ?_, markSieve, ?_, ?_⟩
      · apply SetIteration.branch (flag := false) skipTested (by simpa only [skip, decide_false] using skipTruth)
        apply SetIteration.prepend markedRun
        apply SetIteration.branch (flag := true) (by simpa only [small, decide_true] using smallTest) rfl
        have increment : BlockContinues program env
            [.augAssign (.name "count") .add (.constant (.int 1))] marked nextEnv marked := by
          apply BlockContinues.augAssign (left := number state.2) (right := number 1)
            (value := number (state.2 + 1 : Nat)) (middle := marked) (finalHeap := marked)
            locals.count Evaluates.constant
          simp [inplace, Mutable.binary, scalarBinary, number]
        apply SetIteration.prepend increment
        apply SetIteration.skip
        intro next following
        rfl
      · simpa only [algorithmStep, if_neg skip, if_pos small] using markFound
      · simpa only [algorithmStep, if_neg skip, if_pos small] using locals.updated (state.2 + 1)
    · let stringEnv := setLocal env "s" (.scalar (.string (toString (p : Int))))
      have stringRun : BlockContinues program env [loopBody[3]'(by decide)] marked stringEnv marked := by
        apply BlockContinues.assign
        exact Evaluates.builtin (.cons (Evaluates.lookup getP) .nil) rfl
      have stringLocals := locals.other "s" (.scalar (.string (toString (p : Int)))) (by decide)
      have badTest := bad_characters_evaluates (toString (p : Int)) marked stringEnv rfl stringLocals.allowed
      rw [bad_digits_eq p (by omega)] at badTest
      by_cases allowed : AllowedDigits (ProjectEulerStatements.P35.digitsBE p)
      · let rotsEnv := setLocal stringEnv "rots" (.ref marked.length)
        let rotsHeap := marked ++ [.list (encoded (rotationValues p))]
        have rotsRun : BlockContinues program stringEnv [loopBody[5]'(by decide)] marked rotsEnv rotsHeap :=
          BlockContinues.assign (Evaluates.call (.cons (Evaluates.lookup
            (by simpa [stringEnv, setLocal] using getP)) .nil) (rotations_of.result p (by omega) marked))
        have rotsLocals := stringLocals.other "rots" (.ref marked.length) (by decide)
        have rotsFound : rotsHeap[marked.length]? = some (.list (encoded (rotationValues p))) := by simp [rotsHeap]
        obtain ⟨cycleEnv, cycleRun, unchanged⟩ :=
          mark_rotation_loop limit 0 processedAddress marked.length (rotationValues p) (rotationValues p)
            (insert p state.1) rotsHeap rotsEnv (found_append markFound _) rotsFound
            rotsLocals.processed rotsLocals.limitValue rfl
        let cycleHeap := rotsHeap.set processedAddress
          (.bytearray (processedBytes limit (state.1 ∪ boundedOrbit limit p)))
        have cycleSame : markRotations limit (insert p state.1) (rotationValues p) =
            state.1 ∪ boundedOrbit limit p := markRotations_cycle limit p state.1 (by omega) inside
        rw [cycleSame] at cycleRun
        have cycleLocals := rotsLocals.preserved unchanged
        have cycleFound : cycleHeap[processedAddress]? =
            some (.bytearray (processedBytes limit (state.1 ∪ boundedOrbit limit p))) := by
          apply List.getElem?_set_self
          simp only [rotsHeap, marked, List.length_set, List.length_append]
          omega
        have cycleSieve : cycleHeap[sieveAddress]? = some (.bytearray bytes) := by
          simpa [cycleHeap, List.getElem?_set_ne separate.symm] using found_append markSieve [.list (encoded (rotationValues p))]
        have freshDifferent : processedAddress ≠ marked.length := by simp only [marked, List.length_set]; omega
        have cycleRots : cycleHeap[marked.length]? = some (.list (encoded (rotationValues p))) := by
          simpa [cycleHeap, List.getElem?_set_ne freshDifferent] using rotsFound
        have cycleRotsEnv : cycleEnv "rots" = some (.ref marked.length) := by
          rw [unchanged "rots" (by decide)]
          simp [rotsEnv, setLocal]
        have cycleBlock : BlockContinues program rotsEnv [loopBody[6]'(by decide)] rotsHeap cycleEnv cycleHeap := by
          apply BlockContinues.for_sequence (items := encoded (rotationValues p))
            (by intros; simp) (Evaluates.lookup rfl) (by simp [Mutable.sequence, rotsFound])
          exact cycleRun
        have allTest := all_primes_evaluates (rotationValues p) marked.length sieveAddress bytes cycleHeap cycleEnv
          cycleRots cycleSieve cycleRotsEnv cycleLocals.sieve primes
          (fun n member => by rw [length]; exact rotation_lt_sieve limit p n inside member)
        rw [rotation_primes_eq p lower] at allTest
        cases circular : ProjectEulerStatements.P35.isCircularPrime p with
        | false =>
            refine ⟨cycleHeap, cycleEnv, ?_, cycleSieve, ?_, ?_⟩
            · apply SetIteration.branch (flag := false) skipTested (by simpa only [skip, decide_false] using skipTruth)
              apply SetIteration.prepend markedRun
              apply SetIteration.branch (flag := false) (by simpa only [small, decide_false] using smallTest) rfl
              apply SetIteration.prepend stringRun
              apply SetIteration.branch (flag := false) (by simpa only [allowed, not_true_eq_false, decide_false] using badTest) rfl
              apply SetIteration.prepend rotsRun
              apply SetIteration.prepend cycleBlock
              apply SetIteration.branch (flag := false) (by simpa only [circular] using allTest) rfl
              exact SetIteration.done _ _
            · simpa [algorithmStep, skip, small, allowed, circular] using cycleFound
            · simpa [algorithmStep, skip, small, allowed, circular] using cycleLocals
        | true =>
            let uniqueEnv := setLocal cycleEnv "uniq" (.ref cycleHeap.length)
            let uniqueHeap := cycleHeap ++ [.set (scalarNaturals (rotationValues p).eraseDups)]
            let increment := (boundedOrbit limit p).card
            let finalEnv := setLocal uniqueEnv "count" (number (state.2 + increment : Nat))
            have uniqueRun : BlockContinues program cycleEnv
                [.assign [.name "uniq"] (.builtinCall .set [.name "rots"])] cycleHeap uniqueEnv uniqueHeap := by
              apply BlockContinues.assign
              exact set_rotations_evaluates p marked.length cycleHeap cycleEnv cycleRotsEnv cycleRots
            have uniqueLocals := cycleLocals.other "uniq" (.ref cycleHeap.length) (by decide)
            have sumRun := count_below_evaluates limit cycleHeap.length (rotationValues p).eraseDups
              uniqueHeap uniqueEnv (by simp [uniqueHeap]) rfl uniqueLocals.limitValue
            rw [eraseDups_filter_length p limit] at sumRun
            have incrementRun : SetIteration uniqueEnv
                [.augAssign (.name "count") .add (.sumGenerator (.constant (.int 1))
                  [("r", .name "uniq", [countFilter])] (.constant (.int 0)))] uniqueHeap finalEnv uniqueHeap := by
              apply SetIteration.augAssign (left := number state.2) (right := number increment)
                (value := number (state.2 + increment : Nat)) uniqueLocals.count sumRun
              simp [inplace, Mutable.binary, scalarBinary, number]
            refine ⟨uniqueHeap, finalEnv, ?_, found_append cycleSieve _, ?_, ?_⟩
            · apply SetIteration.branch (flag := false) skipTested (by simpa only [skip, decide_false] using skipTruth)
              apply SetIteration.prepend markedRun
              apply SetIteration.branch (flag := false) (by simpa only [small, decide_false] using smallTest) rfl
              apply SetIteration.prepend stringRun
              apply SetIteration.branch (flag := false) (by simpa only [allowed, not_true_eq_false, decide_false] using badTest) rfl
              apply SetIteration.prepend rotsRun
              apply SetIteration.prepend cycleBlock
              apply SetIteration.branch (flag := true) (by simpa only [circular] using allTest) rfl
              have combined := SetIteration.prepend uniqueRun incrementRun
              simpa only [↓reduceIte, List.append_nil] using combined
            · simpa [algorithmStep, skip, small, allowed] using found_append cycleFound [.set (scalarNaturals (rotationValues p).eraseDups)]
            · simpa [algorithmStep, skip, small, allowed, circular, increment] using uniqueLocals.updated (state.2 + increment)
      · refine ⟨marked, stringEnv, ?_, markSieve, ?_, ?_⟩
        · apply SetIteration.branch (flag := false) skipTested (by simpa only [skip, decide_false] using skipTruth)
          apply SetIteration.prepend markedRun
          apply SetIteration.branch (flag := false) (by simpa only [small, decide_false] using smallTest) rfl
          apply SetIteration.prepend stringRun
          apply SetIteration.branch (flag := true) (by simpa only [allowed, not_false_eq_true, decide_true] using badTest) rfl
          apply SetIteration.skip
          intro next following
          rfl
        · simpa only [algorithmStep, if_neg skip, if_neg small, if_pos allowed] using markFound
        · simpa only [algorithmStep, if_neg skip, if_neg small, if_pos allowed] using stringLocals


private def ScanStep (limit sieveAddress processedAddress : Nat) (bytes : List UInt8) : Prop :=
  ∀ (p : Nat) (state : CountState) (heap : Mutable.Heap) (env : HeapCalls.Env),
    2 ≤ p → p < limit → sieveAddress ≠ processedAddress →
    bytes.length = sieveSize limit →
    (∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i)) →
    heap[sieveAddress]? = some (.bytearray bytes) →
    heap[processedAddress]? = some (.bytearray (processedBytes limit state.1)) →
    ScanLocals env limit sieveAddress processedAddress state.2 →
    env "p" = some (number p) →
    ∃ finalHeap finalEnv, SetIteration env loopBody heap finalEnv finalHeap ∧
      finalHeap[sieveAddress]? = some (.bytearray bytes) ∧
      finalHeap[processedAddress]? = some (.bytearray (processedBytes limit (algorithmStep limit state p).1)) ∧
      ScanLocals finalEnv limit sieveAddress processedAddress (algorithmStep limit state p).2

private theorem count_scan (limit sieveAddress processedAddress : Nat) (bytes : List UInt8)
    (advance : ScanStep limit sieveAddress processedAddress bytes)
    (first count : Nat) (lower : 2 ≤ first) (bound : first + count = limit)
    (state : CountState) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (separate : sieveAddress ≠ processedAddress) (length : bytes.length = sieveSize limit)
    (primes : ∀ i byte, bytes[i]? = some byte → (byte.toNat ≠ 0 ↔ Nat.Prime i))
    (sieveFound : heap[sieveAddress]? = some (.bytearray bytes))
    (processedFound : heap[processedAddress]? = some (.bytearray (processedBytes limit state.1)))
    (locals : ScanLocals env limit sieveAddress processedAddress state.2) :
    ∃ finalHeap, SetCalls.BlockReturns program env
      [.rangeStepLoop "p" first limit 1 loopBody [], .return (.name "count")]
      heap (number ((List.range' first count).foldl (algorithmStep limit) state).2) finalHeap := by
  induction count generalizing first state heap env with
  | zero =>
      refine ⟨heap, SetCalls.BlockReturns.of_heap ?_⟩
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup locals.count)
  | succ count ih =>
      let boundEnv := setLocal env "p" (number first)
      obtain ⟨middle, middleEnv, executed, nextSieve, nextProcessed, nextLocals⟩ :=
        advance first state heap boundEnv lower (by omega) separate length primes sieveFound processedFound
          (locals.other "p" (number first) (by decide)) (by simp [boundEnv, setLocal])
      obtain ⟨finalHeap, remaining⟩ := ih (first + 1) (by omega) (by omega)
        (algorithmStep limit state first) middle middleEnv nextSieve nextProcessed nextLocals
      refine ⟨finalHeap, ?_⟩
      simp only [List.range'_succ, List.foldl_cons]
      intro choices anchors stack used
      let next : Stmt := .rangeStepLoop "p" ((first : Int) + 1) limit 1 loopBody []
      obtain ⟨middleUsed, path⟩ := executed next [.return (.name "count")] choices anchors stack used
      obtain ⟨finalUsed, rest⟩ := remaining choices anchors stack middleUsed
      refine ⟨finalUsed, SetCalls.Reaches.next
        (following := ⟨⟨.block (loopBody ++ [.loopResume next] ++ [.return (.name "count")]) boundEnv,
          stack, heap⟩, used⟩) ?_ ?_⟩
      · simp [SetCalls.step, HeapCalls.step, Calls.rangeContinues,
          show (first : Int) < limit by omega, Calls.loopContinuation,
          show Calls.loopControl loopBody = true from by decide, SetCalls.liftStep,
          boundEnv, next, List.append_assoc]
      · simp only [Nat.cast_add, Nat.cast_one] at rest
        exact path.trans rest

private def outerRangeState (name : String) (env : HeapCalls.Env) (done : List Mutable.Value)
    (expressions : List Expr) (body otherwise following : List Stmt)
    (stack : List Frame) (heap : Mutable.Heap) : HeapCalls.State :=
  match expressions with
  | [] => ⟨.startRange name done body otherwise following env, stack, heap⟩
  | first :: rest =>
      ⟨.expr first env, .rangeArguments name done rest body otherwise following env :: stack, heap⟩

private theorem range_arguments_reach {globals : Mutable.Globals}
    (arguments : Evaluates.Arguments (globals := globals) program env expressions heap values finalHeap) (done : List Mutable.Value)
    (anchors : Nat) (stack : List Frame) :
    HeapCalls.Reaches (globals := globals) program anchors
      (outerRangeState name env done expressions body otherwise following stack heap)
      ⟨.startRange name (done ++ values) body otherwise following env, stack, finalHeap⟩ := by
  induction arguments generalizing done with
  | nil => simp [outerRangeState]; exact .refl
  | @cons expression heap value middle expressions values finalHeap globals head tail ih =>
      apply (head _ _).trans
      refine .next rfl ?_
      simpa [List.append_assoc] using ih (done ++ [value])

private theorem set_for_range
    (arguments : Evaluates.Arguments program env expressions heap values middle)
    (bounds : rangeBounds values = .ok (start, stop, stride))
    (rest : SetCalls.BlockReturns program env
      (.rangeStepLoop name start stop stride body otherwise :: following) middle answer finalHeap) :
    SetCalls.BlockReturns program env
      (.forIn (.name name) (.call (.name "range") expressions) body otherwise :: following)
      heap answer finalHeap := by
  intro choices anchors stack used
  obtain ⟨finalUsed, path⟩ := rest choices anchors stack used
  refine ⟨finalUsed, .next rfl ?_⟩
  apply (SetCalls.Reaches.of_heap (range_arguments_reach arguments [] anchors stack) choices used).trans
  exact .next (by simp [SetCalls.step, HeapCalls.step, bounds, SetCalls.liftStep]) path

private theorem set_branch
    (evaluated : Evaluates program env condition heap value middle)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : SetCalls.BlockReturns program env ((if flag then yes else no) ++ following) middle answer finalHeap) :
    SetCalls.BlockReturns program env (.ifThen condition yes no :: following) heap answer finalHeap := by
  intro choices anchors stack used
  obtain ⟨finalUsed, path⟩ := rest choices anchors stack used
  refine ⟨finalUsed, .next rfl ?_⟩
  apply (SetCalls.Reaches.of_heap (evaluated anchors _) choices used).trans
  exact .next (by simp [SetCalls.step, HeapCalls.step, run, truth, SetCalls.liftStep]) path

private theorem result_of_step (limit : Nat) (heap : Mutable.Heap)
    (advance : ∀ bytes, ScanStep limit (heap.length + 1) (heap.length + 2) bytes) :
    ∃ finalHeap, SetCalls.ReturnsValue program "solve" [number limit] heap
      (number (ProjectEulerStatements.P35.naive limit)) finalHeap := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let initial := functionEnv args body [number limit] []
  have tested : Evaluates program initial (.compare (.name "limit") [(.ltE, .constant (.int 2))]) heap
      (.scalar (.bool (decide (limit ≤ 2)))) heap :=
    Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases small : limit ≤ 2
  · refine ⟨heap, SetCalls.ReturnsValue.of_heap ?_⟩
    rw [naive_small limit small]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := true) (by simpa only [small, decide_true] using tested) rfl
    exact BlockReturns.return_value Evaluates.constant
  · obtain ⟨bytes, initialized, length, primes⟩ := setup limit (by omega) heap initial rfl
    let readyHeap := heap ++ [.bytearray [1], .bytearray bytes, .bytearray (processedBytes limit ∅)]
    let ready := initializedEnv initial heap.length limit
    have locals : ScanLocals ready limit (heap.length + 1) (heap.length + 2) 0 := by
      constructor <;> rfl
    obtain ⟨finalHeap, scanned⟩ := count_scan limit (heap.length + 1) (heap.length + 2) bytes
      (advance bytes) 2 (limit - 2) (by omega) (by omega) (∅, 0) readyHeap ready
      (by omega) length primes (by simp [readyHeap]) (by simp [readyHeap]) locals
    have counted : SetCalls.BlockReturns program ready
        [code[7]'(by decide), code[8]'(by decide)] readyHeap
        (number (ProjectEulerStatements.P35.naive limit)) finalHeap := by
      apply set_for_range (values := [number 2, number limit]) (middle := readyHeap)
        (.cons Evaluates.constant (.cons (Evaluates.lookup locals.limitValue) .nil)) rfl
      have answer : ((List.range' 2 (limit - 2)).foldl (algorithmStep limit) (∅, 0)).2 =
          ProjectEulerStatements.P35.naive limit := algorithmFold_correct limit
      simpa only [answer] using scanned
    refine ⟨finalHeap, ?_⟩
    apply SetCalls.BlockReturns.function rfl rfl
    apply set_branch (flag := false) (by simpa only [small, decide_false] using tested) rfl
    have combined := SetCalls.BlockReturns.after_heap initialized counted
    simpa only [↓reduceIte, List.nil_append] using combined

theorem result (limit : Nat) (heap : Mutable.Heap) :
    ∃ finalHeap, SetCalls.ReturnsValue program "solve" [number limit] heap
      (number (ProjectEulerStatements.P35.naive limit)) finalHeap :=
  result_of_step limit heap (fun bytes => iteration limit (heap.length + 1) (heap.length + 2) bytes)

theorem «Equiv(naive)» (limit : Int)
    (condition : Calls.Condition program
      (fun name => if name = "limit" then some (.int limit) else none)
      Solvers.P35Ast.requires_solve_0) :
    (SetCalls.call program "solve" [number limit] []).Equiv
      (SetCalls.call P35Ast.module "naive" [number limit] []) := by
  have nonnegative : 0 ≤ limit := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
  lift limit to Nat using nonnegative
  intro leftChoices rightChoices
  obtain ⟨leftHeap, optimized⟩ := result limit []
  obtain ⟨rightHeap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p35.naive.result limit []
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [number limit] []).run leftChoices)
    (b := (SetCalls.call P35Ast.module "naive" [number limit] []).run rightChoices)
    (optimized.finishes leftChoices)
    (SetCalls.finishes_of_heap reference.finishes rightChoices 0)
  exact Mutable.Completion.equivalent_allocations [] leftHeap rightHeap
    (number (ProjectEulerStatements.P35.naive limit)) []
    (by simp [Mutable.Heap.Valid]) trivial (by simp)

theorem «Supported» :
    (∀ limit : Int, Calls.Condition program
      (fun name => if name = "limit" then some (.int limit) else none)
      Solvers.P35Ast.requires_solve_0 →
      (SetCalls.call program "solve" [number limit] []).Supported) ∧
    (∀ limit : Int, Calls.Condition program
      (fun name => if name = "limit" then some (.int limit) else none)
      Solvers.P35Ast.requires_solve_0 →
      (SetCalls.call P35Ast.module "naive" [number limit] []).Supported) := by
  constructor
  · intro limit condition
    have nonnegative : 0 ≤ limit := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
    lift limit to Nat using nonnegative
    intro choices
    obtain ⟨_, returned⟩ := result limit []
    exact Execution.supported_of_finishes (returned.finishes choices)
  · intro limit condition
    have nonnegative : 0 ≤ limit := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
    lift limit to Nat using nonnegative
    intro choices
    obtain ⟨_, returned⟩ :=
      data.«project-euler-statements».data.py.project_euler_statements.p35.naive.result limit []
    exact Execution.supported_of_finishes (SetCalls.finishes_of_heap returned.finishes choices 0)

end solve

end solvers.«35»
