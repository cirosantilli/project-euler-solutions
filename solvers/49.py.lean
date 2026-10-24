import «.formalsw».solvers.«49.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p49.py»
import lang.py.FormalSW.HeapCalls
import lang.py.FormalSW.DictionaryItemsCalls
import lang.py.FormalSW.IntegerParsing
import lang.py.FormalSW.SetCalls
import ProjectEulerSolutions.Lib.BooleanPrimeSieve
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.DigitSignatures
import ProjectEulerSolutions.Lib.StringBuckets
import ProjectEulerSolutions.Lib.PrimePermutationSearch
import ProjectEulerSolutions.Lib.PrimeBucketSearch

set_option Elab.async false

namespace solvers.«49»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
private abbrev program := Solvers.P49Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

namespace sieve_primes_upto
open ProjectEulerSolutions.BooleanPrimeSieve ProjectEulerSolutions.PrimeSieve
private def values (flags : List Bool) : List Mutable.Value := flags.map flag
private def code : List Stmt := executableBody ((findFunctionIn program.body "sieve_primes_upto").getD ([], [])).2
private def loopBody : List Stmt := match (code[4]? : Option Stmt) with | some (.whileDo _ body _) => body | _ => []
private def markBody : List Stmt := match (loopBody[0]? : Option Stmt) with | some (.ifThen _ body _) => body | _ => []
private abbrev writeBody : List Stmt := [.assign [.subscript (.name "is_prime") (.name "x")] (.constant (.bool false))]

private theorem heap_inside {heap : Mutable.Heap} {address : Nat}
    (found : heap[address]?=some (object : Mutable.Object)) : address<heap.length :=
  (List.getElem?_eq_some_iff.mp found).1

private theorem heap_set_same (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]?=some object) : heap.set address object=heap := by
  apply List.ext_getElem?
  intro i
  by_cases same : i=address
  · subst i
    rw [List.getElem?_set_self (heap_inside found), found]
  · simp [List.getElem?_set_ne (Ne.symm same)]

private theorem clear_write (index address : Nat) (flags : List Bool) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (inside : index<flags.length) (getIndex : env "x"=some (number index))
    (getArray : env "is_prime"=some (.ref address)) (found : heap[address]?=some (.list (values flags))) :
    BlockContinues program env writeBody heap env (heap.set address (.list (values (flags.set index false)))) := by
  apply BlockContinues.setItem (value := flag false) (by intros; simp) Evaluates.constant
    (.cons (Evaluates.lookup getArray) (.cons (Evaluates.lookup getIndex) .nil))
  have nonnegative : ¬ (index : Int)<0 := by omega
  have bounded : ¬ flags.length≤index := by omega
  simp [Mutable.setItem, Mutable.index, found, values, nonnegative, bounded, Except.map]

/-- The marking range stores False in exactly its live list positions. -/
private theorem clear_loop (first stop step address : Nat) (positive : 0<step) (flags : List Bool)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (size : stop≤flags.length)
    (getArray : env "is_prime"=some (.ref address)) (found : heap[address]?=some (.list (values flags))) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "x" first stop step writeBody []] heap finalEnv
      (heap.set address (.list (values (clearRange flags first stop step)))) ∧
      ∀ query, query ≠ "x" → finalEnv query=env query := by
  induction measure : stop-first using Nat.strong_induction_on generalizing first flags heap env with
  | h size ih =>
      by_cases active : first<stop
      · let bound := setLocal env "x" (number first)
        let middle := heap.set address (.list (values (flags.set first false)))
        have boundArray : bound "is_prime"=some (.ref address) := by simpa [bound, setLocal] using getArray
        have written := clear_write first address flags heap bound (by omega) (by simp [bound, setLocal]) boundArray found
        have nextFound : middle[address]?=some (.list (values (flags.set first false))) :=
          List.getElem?_set_self (heap_inside found)
        obtain ⟨finalEnv, rest, stable⟩ := ih (stop-(first+step)) (by omega) (first+step) (flags.set first false)
          middle bound (by simpa using size) boundArray nextFound rfl
        refine ⟨finalEnv, ?_, ?_⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
          simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_eq_true,
            if_false, Int.natCast_add, clearRange_step flags first stop step positive active, middle, List.set_set] using written.append rest
        · intro query different
          exact (stable query different).trans (by simp [bound, setLocal, different])
      · refine ⟨env, ?_, fun _ _ => rfl⟩
        rw [clearRange_done flags first stop step positive (by omega), heap_set_same heap address _ found]
        exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil

private def markEnv (env : HeapCalls.Env) (p : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "step" (number p)) "start" (number ((p*p : Nat) : Int))

private theorem mark_result (limit p address : Nat) (flags : List Bool) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (lower : 2≤p) (size : flags.length=limit+1)
    (getP : env "p"=some (number p)) (getLimit : env "n"=some (number limit))
    (getArray : env "is_prime"=some (.ref address)) (found : heap[address]?=some (.list (values flags))) :
    ∃ finalEnv, BlockContinues program env markBody heap finalEnv
      (heap.set address (.list (values (ProjectEulerSolutions.BooleanPrimeSieve.mark flags p)))) ∧
      finalEnv "n"=env "n" ∧ finalEnv "is_prime"=env "is_prime" ∧ finalEnv "p"=env "p" := by
  let firstEnv := setLocal env "step" (number p)
  let bound := markEnv env p
  have first : BlockContinues program env [markBody[0]'(by decide)] heap firstEnv heap :=
    BlockContinues.assign (Evaluates.lookup getP)
  have second : BlockContinues program firstEnv [markBody[1]'(by decide)] heap bound heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (Evaluates.lookup (by simpa [firstEnv, setLocal] using getP))
      (Evaluates.lookup (by simpa [firstEnv, setLocal] using getP)) (by simp [Mutable.binary, scalarBinary])
  have getArray' : bound "is_prime"=some (.ref address) := by simpa [bound, markEnv, setLocal] using getArray
  obtain ⟨finalEnv, loop, stable⟩ := clear_loop (p*p) (limit+1) p address (by omega) flags heap bound (by omega) getArray' found
  have third : BlockContinues program bound [markBody[2]'(by decide)] heap finalEnv
      (heap.set address (.list (values (ProjectEulerSolutions.BooleanPrimeSieve.mark flags p)))) := by
    apply BlockContinues.for_range (values := [number ((p*p : Nat) : Int), number ((limit+1 : Nat) : Int), number p])
      (start := (p*p : Nat)) (stop := (limit+1 : Nat)) (stride := p) (middle := heap) (.cons (Evaluates.lookup rfl)
        (.cons (Evaluates.binary (Evaluates.lookup (by simpa [bound, markEnv, setLocal] using getLimit)) Evaluates.constant
          (by simp [Mutable.binary, scalarBinary]))
        (.cons (Evaluates.lookup (by simp [bound, markEnv, setLocal])) .nil)))
      (by
        change (if (p : Int)=0 then Except.error ({ typeName := "ValueError", args := [] } : ExceptionObject)
          else Except.ok (((p*p : Nat) : Int), ((limit+1 : Nat) : Int), (p : Int)))=_
        rw [if_neg (show (p : Int)≠0 by omega)])
    simpa only [ProjectEulerSolutions.BooleanPrimeSieve.mark, size] using loop
  refine ⟨finalEnv, first.append (second.append third), ?_, ?_, ?_⟩
  · simpa [bound, markEnv, setLocal] using stable "n" (by decide)
  · simpa [bound, markEnv, setLocal] using stable "is_prime" (by decide)
  · simpa [bound, markEnv, setLocal] using stable "p" (by decide)

private theorem read_list (heap : Mutable.Heap) (env : HeapCalls.Env) (address p : Nat) (flags : List Bool)
    (inside : p<flags.length) (getP : env "p"=some (number p)) (getArray : env "is_prime"=some (.ref address))
    (found : heap[address]?=some (.list (values flags))) :
    Evaluates program env (.subscript (.name "is_prime") (.name "p")) heap (flag flags[p]) heap := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getArray) (Evaluates.lookup getP)
  simpa only [values, List.getElem_map] using Mutable.getItem_list heap address p (values flags) found (by simpa [values] using inside)

private def Stable (before after : HeapCalls.Env) : Prop :=
  after "n"=before "n" ∧ after "is_prime"=before "is_prime"

private theorem iteration (limit p address : Nat) (flags : List Bool) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (lower : 2≤p) (active : p*p≤limit) (sieved : ProjectEulerSolutions.BooleanPrimeSieve.Sieved limit p flags)
    (getP : env "p"=some (number p)) (getLimit : env "n"=some (number limit))
    (getArray : env "is_prime"=some (.ref address)) (found : heap[address]?=some (.list (values flags))) :
    ∃ finalEnv finalFlags, BlockContinues program env loopBody heap finalEnv
      (heap.set address (.list (values finalFlags))) ∧ ProjectEulerSolutions.BooleanPrimeSieve.Sieved limit (p+1) finalFlags ∧
      finalEnv "p"=some (number ((p+1 : Nat) : Int)) ∧ Stable env finalEnv := by
  have valid : p<flags.length := by have := sieved.1; nlinarith
  have live : flags[p]=true ↔ Nat.Prime p :=
    (sieved.2.2.2 p flags[p] lower (List.getElem?_eq_getElem valid)).trans (survives_current p lower)
  by_cases prime : Nat.Prime p
  · obtain ⟨middleEnv, marked, sameN, sameArray, sameP⟩ := mark_result limit p address flags heap env lower sieved.1 getP getLimit getArray found
    let nextEnv := setLocal middleEnv "p" (number ((p+1 : Nat) : Int))
    refine ⟨nextEnv, ProjectEulerSolutions.BooleanPrimeSieve.mark flags p, ?_,
      ProjectEulerSolutions.BooleanPrimeSieve.sieved_mark limit p flags sieved prime, rfl, ?_⟩
    · have branch : BlockContinues program env [loopBody[0]'(by decide)] heap middleEnv
          (heap.set address (.list (values (ProjectEulerSolutions.BooleanPrimeSieve.mark flags p)))) := by
        apply BlockContinues.branch (flag := true) (read_list heap env address p flags valid getP getArray found)
        · simp [Mutable.truth, scalarTruth, live.mpr prime]
        · exact marked
      apply branch.append
      apply BlockContinues.augAssign (value := number ((p+1 : Nat) : Int))
        (by simpa [sameP] using getP) Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary]
    · simp [Stable, nextEnv, setLocal, sameN, sameArray]
  · let nextEnv := setLocal env "p" (number ((p+1 : Nat) : Int))
    refine ⟨nextEnv, flags, ?_, ProjectEulerSolutions.BooleanPrimeSieve.sieved_skip limit p flags sieved prime, rfl, ?_⟩
    · rw [heap_set_same heap address _ found]
      have branch : BlockContinues program env [loopBody[0]'(by decide)] heap env heap := by
        apply BlockContinues.branch (flag := false) (read_list heap env address p flags valid getP getArray found)
        · have zero : flags[p]=false := Bool.eq_false_iff.mpr (fun same => prime (live.mp same))
          simp [Mutable.truth, scalarTruth, zero]
        · exact .nil
      apply branch.append
      apply BlockContinues.augAssign (value := number ((p+1 : Nat) : Int)) getP Evaluates.constant
      simp [inplace, Mutable.binary, scalarBinary]
    · simp [Stable, nextEnv, setLocal]

private theorem loop (limit p address : Nat) (flags : List Bool) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (lower : 2≤p) (sieved : ProjectEulerSolutions.BooleanPrimeSieve.Sieved limit p flags)
    (getP : env "p"=some (number p)) (getLimit : env "n"=some (number limit))
    (getArray : env "is_prime"=some (.ref address)) (found : heap[address]?=some (.list (values flags))) :
    ∃ finalEnv finalFlags finalP, BlockContinues program env [code[4]'(by decide)] heap finalEnv
      (heap.set address (.list (values finalFlags))) ∧ ProjectEulerSolutions.BooleanPrimeSieve.Sieved limit finalP finalFlags ∧
      limit<finalP^2 ∧ Stable env finalEnv := by
  induction measure : limit+1-p using Nat.strong_induction_on generalizing p flags env heap with
  | h size ih =>
      by_cases active : p*p≤limit
      · obtain ⟨middleEnv, middleFlags, iterated, nextSieved, nextP, stable⟩ :=
          iteration limit p address flags heap env lower active sieved getP getLimit getArray found
        let middle := heap.set address (.list (values middleFlags))
        have nextFound : middle[address]?=some (.list (values middleFlags)) := List.getElem?_set_self (heap_inside found)
        have progress : limit+1-(p+1)<size := by
          have : p≤limit := by nlinarith
          omega
        obtain ⟨finalEnv, finalFlags, finalP, rest, finalSieved, stopped, saved⟩ :=
          ih (limit+1-(p+1)) progress (p+1) middleFlags middle middleEnv
            (by omega) nextSieved nextP (stable.1.trans getLimit) (stable.2.trans getArray) nextFound rfl
        refine ⟨finalEnv, finalFlags, finalP, ?_, finalSieved, stopped, saved.1.trans stable.1, saved.2.trans stable.2⟩
        apply BlockContinues.while_loop (flag := true) (value := flag true) (middle := heap)
        · apply Evaluates.compare (Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP) rfl) (Evaluates.lookup getLimit)
          simp [HeapCalls.compare, scalarCompare, evalCmpOp]
          exact_mod_cast active
        · rfl
        · simpa [loopBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, middle, List.set_set] using iterated.append rest
      · refine ⟨env, flags, p, ?_, sieved, by nlinarith, rfl, rfl⟩
        rw [heap_set_same heap address _ found]
        apply BlockContinues.while_loop (flag := false) (value := flag false) (middle := heap)
        · apply Evaluates.compare (Evaluates.binary (Evaluates.lookup getP) (Evaluates.lookup getP) rfl) (Evaluates.lookup getLimit)
          simp [HeapCalls.compare, scalarCompare, evalCmpOp]
          exact_mod_cast Nat.lt_of_not_ge active
        · rfl
        · exact .nil

private def tableHeap (heap : Mutable.Heap) (flags : List Bool) : Mutable.Heap :=
  heap++[.list [flag true], .list (values flags)]
private def seed (n : Int) : List Bool := List.replicate (n+1).toNat true
private def zeroFlags (n : Int) : List Bool := if 0≤n then (seed n).set 0 false else seed n
private def initialFlags (n : Int) : List Bool := if 1≤n then (zeroFlags n).set 1 false else zeroFlags n
private def initializedEnv (env : HeapCalls.Env) (address : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "is_prime" (.ref address)) "p" (number 2)

private theorem allocate_result (n : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (getN : env "n"=some (number n)) :
    BlockContinues program env [code[0]'(by decide)] heap (setLocal env "is_prime" (.ref (heap.length+1)))
      (tableHeap heap (seed n)) := by
  apply BlockContinues.assign
  apply Evaluates.binary (av := .ref heap.length) (bv := number (n+1))
    (middle := heap++[.list [flag true]]) (afterRight := heap++[.list [flag true]])
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.binary (Evaluates.lookup getN) Evaluates.constant rfl
  · simp [Mutable.binary, Mutable.array, Mutable.allocate, tableHeap, seed, values, Except.map,
      show (BinOp.mult == .mult)=true from rfl]

private theorem initialize_cell (n : Int) (index address : Nat) (flags : List Bool)
    (size : flags.length=(n+1).toNat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n"=some (number n)) (getArray : env "is_prime"=some (.ref address))
    (found : heap[address]?=some (.list (values flags))) :
    BlockContinues program env
      [.ifThen (.compare (.name "n") [(.gtE, .constant (.int index))])
        [.assign [.subscript (.name "is_prime") (.constant (.int index))] (.constant (.bool false))] []]
      heap env (heap.set address (.list (values (if (index : Int)≤n then flags.set index false else flags)))) := by
  have tested : Evaluates program env (.compare (.name "n") [(.gtE, .constant (.int index))])
      heap (flag (decide ((index : Int)≤n))) heap :=
    Evaluates.compare (Evaluates.lookup getN) Evaluates.constant rfl
  by_cases active : (index : Int)≤n
  · rw [if_pos active]
    apply BlockContinues.branch (flag := true) (by simpa only [active, decide_true] using tested) rfl
    apply BlockContinues.setItem (value := flag false) (by intros; simp) Evaluates.constant
      (.cons (Evaluates.lookup getArray) (.cons Evaluates.constant .nil))
    have nonnegative : ¬ (index : Int)<0 := by omega
    have bounded : ¬flags.length≤index := by omega
    simp [Mutable.setItem, Mutable.index, found, values, nonnegative, bounded, Except.map]
  · rw [if_neg active, heap_set_same heap address _ found]
    apply BlockContinues.branch (flag := false) (by simpa only [active, decide_false] using tested) rfl
    exact .nil

/-- The source's guards safely initialize zero and one, including for negative limits. -/
private theorem setup (n : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (getN : env "n"=some (number n)) :
    BlockContinues program env (code.take 4) heap (initializedEnv env (heap.length+1))
      (tableHeap heap (initialFlags n)) := by
  let bound := setLocal env "is_prime" (.ref (heap.length+1))
  have getN' : bound "n"=some (number n) := by simpa [bound, setLocal] using getN
  have getArray : bound "is_prime"=some (.ref (heap.length+1)) := by simp [bound, setLocal]
  have zeroed : BlockContinues program bound [code[1]'(by decide)] (tableHeap heap (seed n)) bound
      (tableHeap heap (zeroFlags n)) := by
    have written := initialize_cell n 0 (heap.length+1) (seed n) (by simp [seed])
      (tableHeap heap (seed n)) bound getN' getArray (by simp [tableHeap])
    simpa [tableHeap, zeroFlags] using written
  have size : (zeroFlags n).length=(n+1).toNat := by
    unfold zeroFlags
    split <;> simp [seed]
  have one : BlockContinues program bound [code[2]'(by decide)] (tableHeap heap (zeroFlags n)) bound
      (tableHeap heap (initialFlags n)) := by
    have written := initialize_cell n 1 (heap.length+1) (zeroFlags n) size
      (tableHeap heap (zeroFlags n)) bound getN' getArray (by simp [tableHeap])
    simpa [tableHeap, initialFlags] using written
  have counter : BlockContinues program bound [code[3]'(by decide)] (tableHeap heap (initialFlags n))
      (initializedEnv env (heap.length+1)) (tableHeap heap (initialFlags n)) := BlockContinues.assign Evaluates.constant
  exact (allocate_result n heap env getN).append (zeroed.append (one.append counter))

private theorem natural_initial (n : Nat) (large : 2≤n) :
    initialFlags n=ProjectEulerSolutions.BooleanPrimeSieve.initial n := by
  have successor : ((n : Int)+1).toNat=n+1 := by omega
  simpa only [initialFlags, zeroFlags, seed, successor, show (0 : Int)≤n by omega,
    show (1 : Int)≤n by omega, if_true] using ProjectEulerSolutions.BooleanPrimeSieve.initial_stores n large

private abbrev initialEnv (n : Int) : HeapCalls.Env :=
  functionEnv ((findFunctionIn program.body "sieve_primes_upto").getD ([], [])).1
    ((findFunctionIn program.body "sieve_primes_upto").getD ([], [])).2 [number n] []

def answer (n : Int) : List Bool := (List.range (n+1).toNat).map (fun p => decide (Nat.Prime p))

/-- The sieve mutates its fresh list and leaves every incoming heap object unchanged. -/
theorem large_result (n : Nat) (large : 2≤n) (heap : Mutable.Heap) :
    ReturnsValue program "sieve_primes_upto" [number n] heap (.ref (heap.length+1)) (tableHeap heap (answer n)) := by
  let initialized := initializedEnv (initialEnv n) (heap.length+1)
  let start := tableHeap heap (ProjectEulerSolutions.BooleanPrimeSieve.initial n)
  have getN : initialized "n"=some (number n) := by simp [initialized, initializedEnv, setLocal]; rfl
  have getP : initialized "p"=some (number 2) := by simp [initialized, initializedEnv, setLocal]
  have getArray : initialized "is_prime"=some (.ref (heap.length+1)) := by simp [initialized, initializedEnv, setLocal]
  have found : start[heap.length+1]?=some (.list (values (ProjectEulerSolutions.BooleanPrimeSieve.initial n))) := by simp [start, tableHeap]
  obtain ⟨finalEnv, flags, p, ran, sieved, stopped, stable⟩ := loop n 2 (heap.length+1)
    (ProjectEulerSolutions.BooleanPrimeSieve.initial n) start initialized (by omega)
    (ProjectEulerSolutions.BooleanPrimeSieve.sieved_initial n large) getP getN getArray found
  have flagsEqual : flags=answer n := by
    simpa only [answer, show ((n : Int)+1).toNat=n+1 by omega] using
      ProjectEulerSolutions.BooleanPrimeSieve.finished n p flags sieved stopped
  have finished : BlockContinues program (initialEnv n) (code.take 5) heap finalEnv (tableHeap heap (answer n)) := by
    have allocated : BlockContinues program (initialEnv n) (code.take 4) heap initialized start := by
      simpa only [natural_initial n large] using setup n heap (initialEnv n) rfl
    have run := allocated.append ran
    simpa [start, tableHeap, flagsEqual] using run
  apply BlockReturns.function rfl rfl
  apply BlockContinues.returns (body := code.take 5) finished
  exact BlockReturns.return_value (Evaluates.lookup (stable.2.trans getArray))

/-- For negative limits and the two small arrays, the while guard is immediately false. -/
theorem small_result (n : Int) (small : n<2) (heap : Mutable.Heap) :
    ReturnsValue program "sieve_primes_upto" [number n] heap (.ref (heap.length+1)) (tableHeap heap (answer n)) := by
  have flagsEqual : initialFlags n=answer n := by
    by_cases negative : n<0
    · have empty : (n+1).toNat=0 := by omega
      simp [initialFlags, zeroFlags, seed, answer, empty, show ¬0≤n by omega, show ¬1≤n by omega]
    · have cases : n=0 ∨ n=1 := by omega
      rcases cases with rfl | rfl <;> decide
  let initialized := initializedEnv (initialEnv n) (heap.length+1)
  have noLoop : BlockReturns program initialized (code.drop 4) (tableHeap heap (initialFlags n))
      (.ref (heap.length+1)) (tableHeap heap (answer n)) := by
    rw [flagsEqual]
    apply BlockReturns.while_loop (flag := false) (value := flag false) (middle := tableHeap heap (answer n))
    · apply Evaluates.compare
        (Evaluates.binary (av := number 2) (bv := number 2) (Evaluates.lookup rfl) (Evaluates.lookup rfl) rfl)
        (Evaluates.lookup (by simp [initialized, initializedEnv, setLocal]; rfl))
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, show ¬4≤n by omega]
    · rfl
    · exact BlockReturns.return_value (Evaluates.lookup (by simp [initialized, initializedEnv, setLocal]))
  apply BlockReturns.function rfl rfl
  simpa only [List.take_append_drop] using (setup n heap (initialEnv n) rfl).returns noLoop

/-- Every signed input returns the inclusive primality table, or an empty list for a negative limit. -/
theorem result (n : Int) (heap : Mutable.Heap) :
    ReturnsValue program "sieve_primes_upto" [number n] heap (.ref (heap.length+1)) (tableHeap heap (answer n)) := by
  by_cases small : n<2
  · exact small_result n small heap
  · lift n to Nat using (by omega : 0≤n)
    exact large_result n (by omega) heap

@[simp] theorem answer_length (n : Nat) : (answer n).length=n+1 := by
  simp [answer, show ((n : Int)+1).toNat=n+1 by omega]

theorem answer_get (n i : Nat) (inside : i≤n) : (answer n)[i]?=some (decide (Nat.Prime i)) := by
  simp [answer, show ((n : Int)+1).toNat=n+1 by omega, show i<n+1 by omega]

theorem «Returns» (n : Int) : (HeapCalls.call program "sieve_primes_upto" [number n] []).Returns :=
  (result n []).returns

theorem «Supported» (n : Int) : (HeapCalls.call program "sieve_primes_upto" [number n] []).Supported :=
  («Returns» n).supported

end sieve_primes_upto

namespace find_prime_perm_arith_sequence
open ProjectEulerSolutions.HeapCollection

private def code : List Stmt :=
  executableBody ((findFunctionIn program.body "find_prime_perm_arith_sequence").getD ([], [])).2
private def groupBody : List Stmt :=
  match (code[3]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private def searchBody : List Stmt :=
  match (code[5]? : Option Stmt) with | some (.forDictionaryItems _ _ body _) => body | _ => []
private def rowBody : List Stmt :=
  match (searchBody[4]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []
private def pairBody : List Stmt :=
  match (searchBody[4]? : Option Stmt) with
  | some (.forIn _ _ (.forIn _ _ body _ :: _) _) => body
  | _ => []
private def pairCondition : Expr :=
  match (pairBody[3]? : Option Stmt) with | some (.ifThen condition _ _) => condition | _ => .constant .none
private def acceptedBody : List Stmt :=
  match (pairBody[3]? : Option Stmt) with | some (.ifThen _ body _) => body | _ => []
private def knownValues : List Mutable.Value := [number 1487, number 4817, number 8147]
private def itemTarget : Expr := .tuple [.name "key", .name "arr"]
private def itemName : String := comprehensionTargetName itemTarget
private def itemBody : List Stmt := comprehensionTargetBody itemTarget searchBody
private def itemKeys (buckets : List ProjectEulerSolutions.StringBuckets.Bucket) : List SequenceSource :=
  (ProjectEulerSolutions.StringBuckets.entries buckets).map (fun entry => .scalar entry.1)
private def itemLoop (dictionary : Nat) (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (position : Nat) : Stmt := .dictionaryItemsStepLoop itemName dictionary (itemKeys buckets) position itemBody []
private def numbers (items : List Nat) : List Mutable.Value := items.map (fun (n : Nat) => number n)
private def primeBody (collector : Nat) : List Stmt :=
  [.ifThen (.subscript (.name "is_prime") (.name "p")) [.comprehensionAppend collector (.name "p")] []]

/-- The comprehension filters the sieve in increasing four-digit order. -/
def primes4d : List Nat := (List.range' 1000 9000).filter (fun p => decide (Nat.Prime p))

private theorem read_prime (heap : Mutable.Heap) (env : HeapCalls.Env) (address limit p : Nat)
    (inside : p≤limit) (getP : env "p"=some (number p))
    (getArray : env "is_prime"=some (.ref address))
    (found : heap[address]?=some (.list (sieve_primes_upto.values (sieve_primes_upto.answer limit)))) :
    Evaluates program env (.subscript (.name "is_prime") (.name "p")) heap
      (flag (decide (Nat.Prime p))) heap := by
  have entry := sieve_primes_upto.answer_get limit p inside
  have mapped : (sieve_primes_upto.values (sieve_primes_upto.answer limit))[p]?=
      some (flag (decide (Nat.Prime p))) := by
    rw [sieve_primes_upto.values, List.getElem?_map, entry]
    rfl
  have size : (sieve_primes_upto.values (sieve_primes_upto.answer limit)).length=limit+1 := by
    rw [sieve_primes_upto.values, List.length_map, sieve_primes_upto.answer_length]
  have valid : p<(sieve_primes_upto.values (sieve_primes_upto.answer limit)).length := by omega
  have live : (sieve_primes_upto.values (sieve_primes_upto.answer limit))[p]=flag (decide (Nat.Prime p)) :=
    Option.some.inj ((List.getElem?_eq_getElem valid).symm.trans mapped)
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro equality; cases equality) (Evaluates.lookup getArray) (Evaluates.lookup getP)
  simpa only [live] using Mutable.getItem_list heap address p _ found valid

private theorem prime_loop (first count : Nat) (stop : first+count=10000)
    (base : Mutable.Heap) (address : Nat)
    (found : base[address]?=some (.list (sieve_primes_upto.values (sieve_primes_upto.answer 9999))))
    (items : List Nat) (env : HeapCalls.Env) (getArray : env "is_prime"=some (.ref address)) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop "p" first 10000 1 (primeBody base.length) []]
      (base++[.list (numbers items)]) finalEnv
      (base++[.list (numbers (items++(List.range' first count).filter (fun p => decide (Nat.Prime p))))]) := by
  induction count generalizing first items env with
  | zero =>
      refine ⟨env, ?_⟩
      simp only [List.range'_zero, List.filter_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env "p" (number first)
      have getArray' : bound "is_prime"=some (.ref address) := by simpa [bound, setLocal] using getArray
      have tested := read_prime (base++[.list (numbers items)]) bound address 9999 first (by omega)
        (by simp [bound, setLocal]) getArray' (prefix_found base _ address _ found)
      have looked : Evaluates program bound (.name "p") (working base (numbers items) [])
          (number first) (working base (numbers items) []) := Evaluates.lookup (by simp [bound, setLocal])
      by_cases prime : Nat.Prime first
      · have emitted : BlockContinues program bound (primeBody base.length)
            (base++[.list (numbers items)]) bound (base++[.list (numbers (items++[first]))]) := by
          apply BlockContinues.branch (flag := true) tested (by simp [prime, Mutable.truth, scalarTruth])
          simpa only [working, numbers, List.map_append, List.map_cons, List.map_nil, List.append_nil] using
            emit base (numbers items) [] [] (number first)
              (by simpa only [List.append_nil] using looked)
        obtain ⟨finalEnv, rest⟩ := ih (first+1) (by omega) (items++[first]) bound getArray'
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, primeBody,
          Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.filter_cons,
          show decide (Nat.Prime first)=true by simp [prime], if_true, Int.natCast_add,
          Int.natCast_one, List.append_assoc, List.singleton_append] using emitted.append rest
      · have skipped : BlockContinues program bound (primeBody base.length)
            (base++[.list (numbers items)]) bound (base++[.list (numbers items)]) := by
          exact BlockContinues.branch (flag := false) tested (by simp [prime, Mutable.truth, scalarTruth]) .nil
        obtain ⟨finalEnv, rest⟩ := ih (first+1) (by omega) items bound getArray'
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, primeBody,
          Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.filter_cons,
          show decide (Nat.Prime first)=false by simp [prime], Int.natCast_add,
          Int.natCast_one] using skipped.append rest

/-- The actual filtered comprehension allocates one list and leaves the sieve intact. -/
theorem primes_result (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (getArray : env "is_prime"=some (.ref address))
    (found : heap[address]?=some (.list (sieve_primes_upto.values (sieve_primes_upto.answer 9999)))) :
    Evaluates program env
      (.listComp (.name "p") [("p", .call (.name "range") [.constant (.int 1000),.constant (.int 10000)],
        [.subscript (.name "is_prime") (.name "p")])]) heap (.ref heap.length)
      (heap++[.list (numbers primes4d)]) := by
  obtain ⟨finalEnv, executed⟩ := prime_loop 1000 9000 rfl heap address found []
    (comprehensionEnv env [("p", .call (.name "range") [.constant (.int 1000),.constant (.int 10000)],
      [.subscript (.name "is_prime") (.name "p")])]) (by simpa [comprehensionEnv] using getArray)
  apply Evaluates.list_comp_range (values := [number 1000,number 10000])
    (start := 1000) (stop := 10000) (stride := 1) (.cons Evaluates.constant (.cons Evaluates.constant .nil)) rfl
  simpa only [comprehensionFilters, comprehensionBody, primeBody, List.nil_append, primes4d] using executed

def preparedHeap (heap : Mutable.Heap) : Mutable.Heap :=
  sieve_primes_upto.tableHeap heap (sieve_primes_upto.answer 9999)++[.list (numbers primes4d)]

def preparedEnv (env : HeapCalls.Env) (heap : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal env "is_prime" (.ref (heap.length+1))) "primes_4d" (.ref (heap.length+2))

/-- The first two original statements construct exactly the four-digit primes. -/
theorem setup_result (heap : Mutable.Heap) (env : HeapCalls.Env) :
    BlockContinues program env (code.take 2) heap (preparedEnv env heap) (preparedHeap heap) := by
  let afterSieve := sieve_primes_upto.tableHeap heap (sieve_primes_upto.answer 9999)
  let bound := setLocal env "is_prime" (.ref (heap.length+1))
  have sieve : BlockContinues program env [code[0]'(by decide)] heap bound afterSieve := by
    apply BlockContinues.assign
    exact Evaluates.call (.cons Evaluates.constant .nil) (sieve_primes_upto.result 9999 heap)
  have selected : BlockContinues program bound [code[1]'(by decide)] afterSieve
      (preparedEnv env heap) (preparedHeap heap) := by
    have evaluated := primes_result afterSieve bound (heap.length+1) (by simp [bound, setLocal])
      (by simp [afterSieve, sieve_primes_upto.tableHeap])
    simpa only [afterSieve, sieve_primes_upto.tableHeap, List.length_append, List.length_cons,
      List.length_nil, Nat.add_zero, preparedHeap, preparedEnv, bound] using BlockContinues.assign evaluated
  exact sieve.append selected

theorem primes4d_mem (p : Nat) : p ∈ primes4d ↔ 1000≤p ∧ p<10000 ∧ Nat.Prime p := by
  simp [primes4d, List.mem_range'_1, and_assoc]

theorem primes4d_ordered : primes4d.Pairwise (· < ·) :=
  (List.pairwise_lt_range' 1 (by decide)).filter _

/-- The original grouping loop computes its digit key and retains its sorted character list. -/
theorem key_result (heap : Mutable.Heap) (env : HeapCalls.Env) (p : Int)
    (getP : env "p"=some (number p)) :
    BlockContinues program env (groupBody.take 1) heap
      (setLocal env "key" (.scalar (.string (ProjectEulerSolutions.DigitSignatures.signature p))))
      (heap++[.list (ProjectEulerSolutions.DigitSignatures.values p)]) :=
  BlockContinues.assign (ProjectEulerSolutions.DigitSignatures.evaluate program env heap "p" p getP)

def keyOf (p : Nat) : String := ProjectEulerSolutions.DigitSignatures.signature p
private def keyHeap (heap : Mutable.Heap) (p : Nat) : Mutable.Heap :=
  heap ++ [.list (ProjectEulerSolutions.DigitSignatures.values p)]
private def groupEnv (env : HeapCalls.Env) (p : Nat) : HeapCalls.Env :=
  setLocal env "key" (.scalar (.string (keyOf p)))

/-- One original iteration retains the character list, eager default, and shared bucket mutation. -/
theorem group_step (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary : Nat)
    (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (view : ProjectEulerSolutions.StringBuckets.View heap dictionary buckets) (p : Nat)
    (getP : env "p" = some (number p)) (getGroups : env "groups" = some (.ref dictionary)) :
    BlockContinues program env groupBody heap (groupEnv env p)
      (ProjectEulerSolutions.StringBuckets.updated (keyHeap heap p) dictionary buckets (keyOf p) p) := by
  have key := key_result heap env p getP
  have inserted := ProjectEulerSolutions.StringBuckets.evaluate (globals := []) program (groupEnv env p)
    (keyHeap heap p) dictionary buckets (view.append _) (keyOf p) p
    (by simpa [groupEnv, setLocal] using getGroups) (by simp [groupEnv, setLocal])
    (by simpa [groupEnv, setLocal, ProjectEulerSolutions.StringBuckets.number] using getP)
  exact key.append (BlockContinues.discard inserted)

private theorem group_prefix (base extra : Mutable.Heap) (dictionary : Nat)
    (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (view : ProjectEulerSolutions.StringBuckets.View (base ++ extra) dictionary buckets)
    (afterBase : base.length ≤ dictionary) (p : Nat) :
    ∃ finalExtra, ProjectEulerSolutions.StringBuckets.updated (keyHeap (base ++ extra) p)
      dictionary buckets (keyOf p) p = base ++ finalExtra := by
  have appended := view.append [.list (ProjectEulerSolutions.DigitSignatures.values p)]
  obtain ⟨finalExtra, extended⟩ := ProjectEulerSolutions.StringBuckets.updated_prefix base
    (extra ++ [.list (ProjectEulerSolutions.DigitSignatures.values p)]) dictionary buckets
    (by simpa only [List.append_assoc] using appended) afterBase (keyOf p) p
  exact ⟨finalExtra, by simpa only [keyHeap, List.append_assoc] using extended⟩

private theorem groupBody_control : Calls.loopControl groupBody = false := rfl

/-- The source loop constructs the exact stable partition while preserving its input list. -/
private theorem group_loop (base : Mutable.Heap) (input : Nat) (items : List Nat)
    (found : base[input]? = some (.list (numbers items))) (dictionary : Nat)
    (afterBase : base.length ≤ dictionary) (remaining : List Nat) (position : Nat)
    (remainingEq : items.drop position = remaining) (processed : List Nat) (extra : Mutable.Heap)
    (env : HeapCalls.Env) (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (view : ProjectEulerSolutions.StringBuckets.View (base ++ extra) dictionary buckets)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf processed buckets)
    (getGroups : env "groups" = some (.ref dictionary)) :
    ∃ finalEnv finalExtra finalBuckets,
      BlockContinues program env [.sequenceStepLoop "p" (.ref input) position groupBody []]
        (base ++ extra) finalEnv (base ++ finalExtra) ∧
      ProjectEulerSolutions.StringBuckets.View (base ++ finalExtra) dictionary finalBuckets ∧
      ProjectEulerSolutions.StringBuckets.Partition keyOf (processed ++ remaining) finalBuckets ∧
      finalEnv "groups" = some (.ref dictionary) := by
  induction remaining generalizing position processed extra env buckets with
  | nil =>
      refine ⟨env, extra, buckets, ?_, view, ?_, getGroups⟩
      · apply BlockContinues.sequence_done (items := numbers items)
          (by simp [sequenceValue, Mutable.sequence, prefix_found base extra input _ found])
        · have missing : items[position]? = none := by
            simpa [remainingEq] using (List.head?_drop (l := items) (i := position)).symm
          simp [numbers, missing]
        · exact .nil
      · simpa only [List.append_nil] using partition
  | cons head tail ih =>
      let bound := setLocal env "p" (number head)
      let ready := groupEnv bound head
      let afterKey := keyHeap (base ++ extra) head
      let nextBuckets := ProjectEulerSolutions.StringBuckets.push buckets (keyOf head) afterKey.length head
      have selected : items[position]? = some head := by
        simpa [remainingEq] using (List.head?_drop (l := items) (i := position)).symm
      have executed := group_step (base ++ extra) bound dictionary buckets view head
        (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getGroups)
      obtain ⟨nextExtra, extended⟩ := group_prefix base extra dictionary buckets view afterBase head
      have nextView : ProjectEulerSolutions.StringBuckets.View (base ++ nextExtra) dictionary nextBuckets := by
        rw [← extended]
        exact ProjectEulerSolutions.StringBuckets.updated_view afterKey dictionary buckets (view.append _) (keyOf head) head
      have nextPartition : ProjectEulerSolutions.StringBuckets.Partition keyOf (processed ++ [head]) nextBuckets :=
        partition.push afterKey.length head
      obtain ⟨finalEnv, finalExtra, finalBuckets, rest, finalView, finalPartition, finalGroups⟩ :=
        ih (position := position + 1)
          (remainingEq := by simp [List.drop_add_one_eq_tail_drop, remainingEq])
          (processed := processed ++ [head]) (extra := nextExtra) (env := ready) (buckets := nextBuckets)
          nextView nextPartition (by simpa [ready, groupEnv, bound, setLocal] using getGroups)
      refine ⟨finalEnv, finalExtra, finalBuckets, ?_, finalView, ?_, finalGroups⟩
      · apply BlockContinues.sequence_step (items := numbers items) (value := number head)
          (by simp [sequenceValue, Mutable.sequence, prefix_found base extra input _ found])
          (by simp [numbers, selected])
        rw [extended] at executed
        simpa only [Calls.loopContinuation, groupBody_control, Bool.false_eq_true, if_false] using executed.append rest
      · simpa only [List.append_assoc, List.singleton_append] using finalPartition

/-- The original setup and grouping statements build exactly the four-digit prime buckets. -/
theorem groups_result (heap : Mutable.Heap) (env : HeapCalls.Env) :
    ∃ finalEnv extra buckets,
      BlockContinues program env (code.take 4) heap finalEnv (preparedHeap heap ++ extra) ∧
      ProjectEulerSolutions.StringBuckets.View (preparedHeap heap ++ extra) (preparedHeap heap).length buckets ∧
      ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets ∧
      finalEnv "groups" = some (.ref (preparedHeap heap).length) := by
  let base := preparedHeap heap
  let bound := setLocal (preparedEnv env heap) "groups" (.ref base.length)
  have inputFound : base[heap.length + 2]? = some (.list (numbers primes4d)) := by
    simp [base, preparedHeap, sieve_primes_upto.tableHeap]
  have allocated : BlockContinues program (preparedEnv env heap) [code[2]'(by decide)] base bound
      (base ++ [.dict []]) := BlockContinues.annAssign (Evaluates.dictionary .nil)
  obtain ⟨finalEnv, extra, buckets, executed, view, partition, finalGroups⟩ :=
    group_loop base (heap.length + 2) primes4d inputFound base.length (by omega)
      primes4d 0 rfl [] [.dict []] bound [] (ProjectEulerSolutions.StringBuckets.empty_view base)
      (ProjectEulerSolutions.StringBuckets.Partition.empty keyOf) (by simp [bound, setLocal])
  have iterated : BlockContinues program bound [code[3]'(by decide)] (base ++ [.dict []]) finalEnv
      (base ++ extra) := by
    apply BlockContinues.for_sequence (value := .ref (heap.length + 2)) (items := numbers primes4d)
      (by intros; intro impossible; cases impossible)
      (Evaluates.lookup (by simp [bound, preparedEnv, setLocal]))
      (by simp [Mutable.sequence, prefix_found base [.dict []] (heap.length + 2) _ inputFound])
      executed
  exact ⟨finalEnv, extra, buckets, (setup_result heap env).append (allocated.append iterated), view,
    by simpa only [List.nil_append] using partition, finalGroups⟩

theorem known_result (heap : Mutable.Heap) (env : HeapCalls.Env) :
    BlockContinues program env [code[4]'(by decide)] heap
      (setLocal env "target_known" (.ref heap.length)) (heap ++ [.tuple knownValues]) := by
  apply BlockContinues.assign
  exact Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))
    (by simp [knownValues])

/-- The dictionary scan receives the actual shared buckets and freshly allocated excluded example. -/
theorem search_ready_result (heap : Mutable.Heap) (env : HeapCalls.Env) :
    ∃ finalEnv extra buckets knownAddress,
      BlockContinues program env (code.take 5) heap finalEnv (preparedHeap heap ++ extra) ∧
      ProjectEulerSolutions.StringBuckets.View (preparedHeap heap ++ extra) (preparedHeap heap).length buckets ∧
      ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets ∧
      finalEnv "groups" = some (.ref (preparedHeap heap).length) ∧
      finalEnv "target_known" = some (.ref knownAddress) ∧
      (preparedHeap heap ++ extra)[knownAddress]? = some (.tuple knownValues) := by
  obtain ⟨groupedEnv, extra, buckets, grouped, view, partition, getGroups⟩ := groups_result heap env
  let base := preparedHeap heap ++ extra
  let ready := setLocal groupedEnv "target_known" (.ref base.length)
  refine ⟨ready, extra ++ [.tuple knownValues], buckets, base.length, ?_, ?_, partition, ?_, ?_, ?_⟩
  · simpa only [base, List.append_assoc] using grouped.append (known_result base groupedEnv)
  · simpa only [List.append_assoc] using view.append [.tuple knownValues]
  · simpa [ready, setLocal] using getGroups
  · simp [ready, setLocal]
  · rw [← List.append_assoc]
    simp [base]

theorem bucket_member (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets)
    (bucket : ProjectEulerSolutions.StringBuckets.Bucket) (member : bucket ∈ buckets)
    (p : Nat) (present : p ∈ bucket.values) :
    1000 ≤ p ∧ p < 10000 ∧ Nat.Prime p ∧ keyOf p = bucket.key := by
  rw [partition.bucket_values bucket member] at present
  obtain ⟨prime, key⟩ := List.mem_filter.mp present
  obtain ⟨lower, upper, primality⟩ := (primes4d_mem p).mp prime
  exact ⟨lower, upper, primality, of_decide_eq_true key⟩

theorem bucket_ordered (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets)
    (bucket : ProjectEulerSolutions.StringBuckets.Bucket) (member : bucket ∈ buckets) :
    bucket.values.Pairwise (· < ·) := by
  rw [partition.bucket_values bucket member]
  exact primes4d_ordered.filter _

/-- Any increasing arithmetic triple drawn from a constructed bucket is one of the two certified rows. -/
theorem bucket_classifies (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets)
    (bucket : ProjectEulerSolutions.StringBuckets.Bucket) (member : bucket ∈ buckets)
    (a b c : Nat) (aMember : a ∈ bucket.values) (bMember : b ∈ bucket.values) (cMember : c ∈ bucket.values)
    (ab : a < b) (bc : b < c) (arithmetic : c = 2 * b - a) :
    (a,b,c) = ProjectEulerSolutions.PrimePermutationGroups.known ∨
      (a,b,c) = ProjectEulerSolutions.PrimePermutationGroups.winner := by
  have ha := bucket_member buckets partition bucket member a aMember
  have hb := bucket_member buckets partition bucket member b bMember
  have hc := bucket_member buckets partition bucket member c cMember
  have digitPermutation (p : Nat) (positive : 0 < p) (same : keyOf a = keyOf p) :
      ProjectEulerStatements.P49.isPermutation a p = true := by
    rw [← ProjectEulerSolutions.PrimePermutations.permutation_statement]
    apply decide_eq_true
    apply (ProjectEulerSolutions.PrimePermutations.sortedDigits_eq_iff a p).mpr
    exact ProjectEulerSolutions.DigitSignatures.equal_implies_permutation a p (by omega) positive same
  exact ProjectEulerSolutions.PrimePermutationSearch.classifies a b c ha.1 hc.2.1 ab bc arithmetic
    ha.2.2.1 hb.2.2.1 hc.2.2.1
    (digitPermutation b (by omega) (ha.2.2.2.trans hb.2.2.2.symm))
    (digitPermutation c (by omega) (ha.2.2.2.trans hc.2.2.2.symm))

/-- All three numbers of the required progression occur together in a constructed bucket. -/
theorem winner_bucket (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets) :
    ∃ bucket ∈ buckets, 2969 ∈ bucket.values ∧ 6299 ∈ bucket.values ∧ 9629 ∈ bucket.values := by
  have firstPrime : Nat.Prime 2969 := by
    simpa only [ProjectEulerSolutions.SixStepPrimality.certifiedPrime_correct, decide_eq_true_eq] using
      (show ProjectEulerSolutions.SixStepPrimality.certifiedPrime 2969 = true by decide +kernel)
  have secondPrime : Nat.Prime 6299 := by
    simpa only [ProjectEulerSolutions.SixStepPrimality.certifiedPrime_correct, decide_eq_true_eq] using
      (show ProjectEulerSolutions.SixStepPrimality.certifiedPrime 6299 = true by decide +kernel)
  have thirdPrime : Nat.Prime 9629 := by
    simpa only [ProjectEulerSolutions.SixStepPrimality.certifiedPrime_correct, decide_eq_true_eq] using
      (show ProjectEulerSolutions.SixStepPrimality.certifiedPrime 9629 = true by decide +kernel)
  have firstMember : 2969 ∈ primes4d := (primes4d_mem 2969).mpr ⟨by decide, by decide, firstPrime⟩
  have secondMember : 6299 ∈ primes4d := (primes4d_mem 6299).mpr ⟨by decide, by decide, secondPrime⟩
  have thirdMember : 9629 ∈ primes4d := (primes4d_mem 9629).mpr ⟨by decide, by decide, thirdPrime⟩
  obtain ⟨bucket, member, key, first⟩ := partition.member 2969 firstMember
  refine ⟨bucket, member, first, ?_, ?_⟩
  · rw [partition.bucket_values bucket member]
    exact List.mem_filter.mpr ⟨secondMember, by
      apply decide_eq_true
      exact ProjectEulerSolutions.DigitSignatures.winner_signatures.1.symm.trans key.symm⟩
  · rw [partition.bucket_values bucket member]
    exact List.mem_filter.mpr ⟨thirdMember, by
      apply decide_eq_true
      exact ProjectEulerSolutions.DigitSignatures.winner_signatures.2.symm.trans key.symm⟩

theorem bucket_classification (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets)
    (bucket : ProjectEulerSolutions.StringBuckets.Bucket) (member : bucket ∈ buckets) :
    ProjectEulerSolutions.PrimeBucketSearch.Classifies bucket.values := by
  intro a aMember b bMember ab cMember
  exact bucket_classifies buckets partition bucket member a b (ProjectEulerSolutions.PrimeBucketSearch.third a b)
    aMember bMember cMember ab (by simp only [ProjectEulerSolutions.PrimeBucketSearch.third]; omega)
    (by simp only [ProjectEulerSolutions.PrimeBucketSearch.third]; omega)

/-- A successful native pair scan of any constructed bucket can return only the required row. -/
theorem bucket_search_sound (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets)
    (bucket : ProjectEulerSolutions.StringBuckets.Bucket) (member : bucket ∈ buckets)
    (answer : ProjectEulerSolutions.PrimePermutationGroups.Row)
    (result : ProjectEulerSolutions.PrimeBucketSearch.pairs bucket.values bucket.values = some answer) :
    answer = ProjectEulerSolutions.PrimePermutationGroups.winner :=
  ProjectEulerSolutions.PrimeBucketSearch.pairs_sound bucket.values bucket.values (fun _ inside => inside)
    (bucket_ordered buckets partition bucket member) (bucket_classification buckets partition bucket member) answer result

theorem winner_search (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets) :
    ∃ bucket ∈ buckets, ProjectEulerSolutions.PrimeBucketSearch.pairs bucket.values bucket.values =
      some ProjectEulerSolutions.PrimePermutationGroups.winner := by
  obtain ⟨bucket, member, first, second, last⟩ := winner_bucket buckets partition
  exact ⟨bucket, member, ProjectEulerSolutions.PrimeBucketSearch.complete bucket.values
    (bucket_ordered buckets partition bucket member) (bucket_classification buckets partition bucket member)
    first second last⟩

/-- For a nontrivial ordered bucket, the source prepares its set and bounds without changing the list. -/
theorem bucket_setup (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat) (items : List Nat)
    (getArray : env "arr" = some (.ref address)) (found : heap[address]? = some (.list (numbers items)))
    (ordered : items.Pairwise (· < ·)) (large : 3 ≤ items.length) :
    BlockContinues program env (searchBody.take 4) heap
      (setLocal (setLocal env "s" (.ref heap.length)) "m" (number items.length))
      (heap ++ [.set (ProjectEulerSolutions.PrimeBucketSearch.setKeys items)]) := by
  let afterSet := heap ++ [Mutable.Object.set (ProjectEulerSolutions.PrimeBucketSearch.setKeys items)]
  let setEnv := setLocal env "s" (.ref heap.length)
  have shared : heap[address]? = some (.list (ProjectEulerSolutions.PrimeBucketSearch.numbers items)) := by
    simpa only [ProjectEulerSolutions.PrimeBucketSearch.numbers, ProjectEulerSolutions.PrimeBucketSearch.scalarKeys,
      numbers, List.map_map, Function.comp_def] using found
  have sized : Evaluates program env (.builtinCall .len [.name "arr"]) heap (number items.length) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getArray) .nil)
    simp [primitive, Mutable.length, found, numbers]
  have checked : BlockContinues program env [searchBody[0]'(by decide)] heap env heap := by
    apply BlockContinues.branch (flag := false) (value := flag false) (middle := heap)
    · apply Evaluates.compare (av := number items.length) (bv := number 3)
        (middle := heap) (afterRight := heap) sized Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, show ¬ (items.length : Int) < 3 by omega]
    · rfl
    · exact .nil
  have sorted : BlockContinues program env [searchBody[1]'(by decide)] heap env heap := by
    apply BlockContinues.discard
    apply Evaluates.sort (Evaluates.lookup getArray) shared
      (ProjectEulerSolutions.PrimeBucketSearch.sort_ordered heap address items shared
        (ordered.imp Nat.le_of_lt))
    intro name same
    cases same
    simp [getArray]
  have built : BlockContinues program env [searchBody[2]'(by decide)] heap setEnv afterSet := by
    apply BlockContinues.assign
    exact Evaluates.builtin (.cons (Evaluates.lookup getArray) .nil)
      (ProjectEulerSolutions.PrimeBucketSearch.set_from_numbers heap address items shared)
  have bounded : BlockContinues program setEnv [searchBody[3]'(by decide)] afterSet
      (setLocal setEnv "m" (number items.length)) afterSet := by
    apply BlockContinues.assign (value := number items.length)
    apply Evaluates.builtin (values := [.ref address])
      (.cons (Evaluates.lookup (by simpa [setEnv, setLocal] using getArray)) .nil)
    simp [primitive, Mutable.length, afterSet, prefix_found heap _ address _ found, numbers]
  exact checked.append (sorted.append (built.append bounded))

private theorem read_number (heap : Mutable.Heap) (env : HeapCalls.Env) (address position value : Nat)
    (items : List Nat) (name : String) (getArray : env "arr" = some (.ref address))
    (getIndex : env name = some (number position)) (found : heap[address]? = some (.list (numbers items)))
    (selected : items[position]? = some value) :
    Evaluates program env (.subscript (.name "arr") (.name name)) heap (number value) heap := by
  have bound : position < (numbers items).length := by
    simpa only [numbers, List.length_map] using (List.getElem?_eq_some_iff.mp selected).1
  have mapped : (numbers items)[position]? = some (number value) := by simp [numbers, selected]
  have item : (numbers items)[position] = number value :=
    Option.some.inj ((List.getElem?_eq_getElem bound).symm.trans mapped)
  apply Evaluates.subscript (middle := heap) (afterIndex := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getArray) (Evaluates.lookup getIndex)
  simpa only [item] using Mutable.getItem_list heap address position (numbers items) found bound

private def pairEnv (env : HeapCalls.Env) (a b : Nat) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal env "a" (number a)) "b" (number b))
    "d" (number (b - a : Nat))) "c" (number (ProjectEulerSolutions.PrimeBucketSearch.third a b))

/-- Each original pair iteration allocates its unpacked tuple before computing the signed difference and next term. -/
theorem pair_prefix (heap : Mutable.Heap) (env : HeapCalls.Env) (address i j a b : Nat) (items : List Nat)
    (getArray : env "arr" = some (.ref address)) (getI : env "i" = some (number i))
    (getJ : env "j" = some (number j)) (found : heap[address]? = some (.list (numbers items)))
    (first : items[i]? = some a) (second : items[j]? = some b) (increasing : a ≤ b) :
    BlockContinues program env (pairBody.take 3) heap (pairEnv env a b)
      (heap ++ [.tuple [number a, number b]]) := by
  let afterTuple := heap ++ [Mutable.Object.tuple [number a, number b]]
  let unpacked := setLocal (setLocal env "a" (number a)) "b" (number b)
  let afterDifference := setLocal unpacked "d" (number (b - a : Nat))
  have unpackedPair : BlockContinues program env [pairBody[0]'(by decide)] heap unpacked afterTuple := by
    apply BlockContinues.unpack (names := ["a", "b"]) (items := [number a, number b])
      (middle := afterTuple) rfl
    · exact Evaluates.tuple
        (.cons (read_number heap env address i a items "i" getArray getI found first)
          (.cons (read_number heap env address j b items "j" getArray getJ found second) .nil))
        (by simp)
    · simp [Mutable.sequence, afterTuple]
    · rfl
  have difference : (b : Int) - a = ((b - a : Nat) : Int) := by omega
  have subtracted : BlockContinues program unpacked [pairBody[1]'(by decide)] afterTuple afterDifference afterTuple := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number b) (bv := number a) (middle := afterTuple) (afterRight := afterTuple)
      (Evaluates.lookup (by simp [unpacked, setLocal]))
      (Evaluates.lookup (by simp [unpacked, setLocal])) (by simp [Mutable.binary, scalarBinary, difference])
  have advanced : BlockContinues program afterDifference [pairBody[2]'(by decide)] afterTuple
      (pairEnv env a b) afterTuple := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number b) (bv := number (b - a : Nat))
      (middle := afterTuple) (afterRight := afterTuple)
      (Evaluates.lookup (by simp [afterDifference, unpacked, setLocal]))
      (Evaluates.lookup (by simp [afterDifference, setLocal]))
      (by simp [Mutable.binary, scalarBinary, ProjectEulerSolutions.PrimeBucketSearch.third])
  exact unpackedPair.append (subtracted.append advanced)

theorem candidate_present (heap : Mutable.Heap) (env : HeapCalls.Env) (address value : Nat) (items : List Nat)
    (getValue : env "c" = some (number value)) (getSet : env "s" = some (.ref address))
    (found : heap[address]? = some (.set (ProjectEulerSolutions.PrimeBucketSearch.setKeys items))) :
    Evaluates program env pairCondition heap (flag (decide (value ∈ items))) heap := by
  apply Evaluates.membership (.cons (Evaluates.lookup getValue) (.cons (Evaluates.lookup getSet) .nil))
  simp [HeapCalls.membership, found, ProjectEulerSolutions.PrimeBucketSearch.set_contains]

theorem candidate_result (heap : Mutable.Heap) (env : HeapCalls.Env) (a b c : Nat)
    (getA : env "a" = some (number a)) (getB : env "b" = some (number b))
    (getC : env "c" = some (number c)) :
    BlockContinues program env (acceptedBody.take 1) heap (setLocal env "seq" (.ref heap.length))
      (heap ++ [.tuple [number a, number b, number c]]) := by
  apply BlockContinues.assign
  exact Evaluates.tuple (.cons (Evaluates.lookup getA) (.cons (Evaluates.lookup getB)
    (.cons (Evaluates.lookup getC) .nil))) (by simp)

/-- Comparing the freshly allocated candidate with the stored example tests their integer contents. -/
theorem known_comparison (heap : Mutable.Heap) (env : HeapCalls.Env) (address knownAddress a b c : Nat)
    (getCandidate : env "seq" = some (.ref address)) (getKnown : env "target_known" = some (.ref knownAddress))
    (candidate : heap[address]? = some (.tuple [number a, number b, number c]))
    (known : heap[knownAddress]? = some (.tuple knownValues)) (different : address ≠ knownAddress) :
    Evaluates program env (.compare (.name "seq") [(.notEq, .name "target_known")]) heap
      (flag (decide ((a,b,c) ≠ ProjectEulerSolutions.PrimePermutationGroups.known))) heap := by
  apply Evaluates.compare (av := .ref address) (bv := .ref knownAddress)
    (middle := heap) (afterRight := heap) (Evaluates.lookup getCandidate) (Evaluates.lookup getKnown)
  have aTest : (a : Int) = 1487 ↔ a = 1487 := by omega
  have bTest : (b : Int) = 4817 ↔ b = 4817 := by omega
  have cTest : (c : Int) = 8147 ↔ c = 8147 := by omega
  simp [HeapCalls.compare, candidate, known, different, knownValues, equalLists, equalElements,
    scalarCompare, evalCmpOp, ProjectEulerSolutions.PrimePermutationGroups.known, Prod.ext_iff,
    aTest, bTest, cTest, show (CmpOp.notEq == CmpOp.eq) = false by rfl]
  split_ifs <;> simp_all

private def rowValues (row : ProjectEulerSolutions.PrimePermutationGroups.Row) : List Mutable.Value :=
  [number row.1, number row.2.1, number row.2.2]

private structure SearchHeap (heap : Mutable.Heap) (arrayAddress setAddress knownAddress : Nat)
    (items : List Nat) : Prop where
  arrayFound : heap[arrayAddress]? = some (.list (numbers items))
  setFound : heap[setAddress]? = some (.set (ProjectEulerSolutions.PrimeBucketSearch.setKeys items))
  knownFound : heap[knownAddress]? = some (.tuple knownValues)

private theorem SearchHeap.append (snapshot : SearchHeap heap arrayAddress setAddress knownAddress items)
    (extra : Mutable.Heap) : SearchHeap (heap ++ extra) arrayAddress setAddress knownAddress items :=
  ⟨prefix_found heap extra _ _ snapshot.arrayFound, prefix_found heap extra _ _ snapshot.setFound,
    prefix_found heap extra _ _ snapshot.knownFound⟩

private structure SearchContext (env : HeapCalls.Env) (arrayAddress setAddress knownAddress : Nat)
    (items : List Nat) : Prop where
  getArray : env "arr" = some (.ref arrayAddress)
  getSet : env "s" = some (.ref setAddress)
  getLength : env "m" = some (number items.length)
  getKnown : env "target_known" = some (.ref knownAddress)

private theorem SearchContext.pair (context : SearchContext env arrayAddress setAddress knownAddress items)
    (a b : Nat) : SearchContext (pairEnv env a b) arrayAddress setAddress knownAddress items :=
  ⟨by simpa [pairEnv, setLocal] using context.getArray,
    by simpa [pairEnv, setLocal] using context.getSet,
    by simpa [pairEnv, setLocal] using context.getLength,
    by simpa [pairEnv, setLocal] using context.getKnown⟩

private theorem SearchContext.seq (context : SearchContext env arrayAddress setAddress knownAddress items)
    (address : Nat) : SearchContext (setLocal env "seq" (.ref address)) arrayAddress setAddress knownAddress items :=
  ⟨by simpa [setLocal] using context.getArray, by simpa [setLocal] using context.getSet,
    by simpa [setLocal] using context.getLength, by simpa [setLocal] using context.getKnown⟩

private theorem SearchContext.bind_i (context : SearchContext env arrayAddress setAddress knownAddress items)
    (i : Nat) : SearchContext (setLocal env "i" (number i)) arrayAddress setAddress knownAddress items :=
  ⟨by simpa [setLocal] using context.getArray, by simpa [setLocal] using context.getSet,
    by simpa [setLocal] using context.getLength, by simpa [setLocal] using context.getKnown⟩

private theorem SearchContext.bind_j (context : SearchContext env arrayAddress setAddress knownAddress items)
    (j : Nat) : SearchContext (setLocal env "j" (number j)) arrayAddress setAddress knownAddress items :=
  ⟨by simpa [setLocal] using context.getArray, by simpa [setLocal] using context.getSet,
    by simpa [setLocal] using context.getLength, by simpa [setLocal] using context.getKnown⟩

private theorem fresh_comparison (heap : Mutable.Heap) (env : HeapCalls.Env) (knownAddress a b c : Nat)
    (getKnown : env "target_known" = some (.ref knownAddress))
    (known : heap[knownAddress]? = some (.tuple knownValues)) :
    Evaluates program (setLocal env "seq" (.ref heap.length))
      (.compare (.name "seq") [(.notEq, .name "target_known")])
      (heap ++ [.tuple [number a, number b, number c]])
      (flag (decide ((a,b,c) ≠ ProjectEulerSolutions.PrimePermutationGroups.known)))
      (heap ++ [.tuple [number a, number b, number c]]) := by
  apply known_comparison _ _ heap.length knownAddress a b c (by simp [setLocal])
    (by simpa [setLocal] using getKnown) (by simp) (prefix_found heap _ _ _ known)
  have bound := (List.getElem?_eq_some_iff.mp known).1
  omega

/-- A successful original pair iteration returns its freshly allocated candidate tuple. -/
private theorem pair_return (heap : Mutable.Heap) (env : HeapCalls.Env) (arrayAddress setAddress knownAddress i j a b : Nat)
    (items : List Nat) (snapshot : SearchHeap heap arrayAddress setAddress knownAddress items)
    (context : SearchContext env arrayAddress setAddress knownAddress items)
    (getI : env "i" = some (number i)) (getJ : env "j" = some (number j))
    (first : items[i]? = some a) (second : items[j]? = some b) (increasing : a ≤ b)
    (accepted : ProjectEulerSolutions.PrimeBucketSearch.accepted items a b = true) (following : List Stmt) :
    BlockReturns program env (pairBody ++ following) heap (.ref ((heap ++ [Mutable.Object.tuple [number a, number b]]).length))
      (heap ++ [.tuple [number a, number b]] ++
        [.tuple (rowValues (ProjectEulerSolutions.PrimeBucketSearch.candidate a b))]) := by
  let c := ProjectEulerSolutions.PrimeBucketSearch.third a b
  let afterPair := pairEnv env a b
  let firstHeap := heap ++ [Mutable.Object.tuple [number a, number b]]
  have valid : c ∈ items ∧ (a,b,c) ≠ ProjectEulerSolutions.PrimePermutationGroups.known := of_decide_eq_true accepted
  have locals := context.pair a b
  have extended := snapshot.append [Mutable.Object.tuple [number a, number b]]
  have memberTest := candidate_present firstHeap afterPair setAddress c items
    (by simp [afterPair, pairEnv, setLocal, c]) locals.getSet extended.setFound
  have allocated := candidate_result firstHeap afterPair a b c
    (by simp [afterPair, pairEnv, setLocal]) (by simp [afterPair, pairEnv, setLocal])
    (by simp [afterPair, pairEnv, setLocal, c])
  have compared := fresh_comparison firstHeap afterPair knownAddress a b c locals.getKnown extended.knownFound
  apply (pair_prefix heap env arrayAddress i j a b items context.getArray getI getJ snapshot.arrayFound first second increasing).returns
  apply BlockReturns.branch (flag := true) (middle := firstHeap)
    memberTest (by simp [valid.1, Mutable.truth, scalarTruth])
  apply allocated.returns
  apply BlockReturns.branch (flag := true)
    compared (by simp [valid.2, Mutable.truth, scalarTruth])
  exact BlockReturns.return_value (Evaluates.lookup (by simp [setLocal, firstHeap]))

/-- A rejected pair retains all allocations and preserves the captured search inputs. -/
private theorem pair_skip (heap : Mutable.Heap) (env : HeapCalls.Env) (arrayAddress setAddress knownAddress i j a b : Nat)
    (items : List Nat) (snapshot : SearchHeap heap arrayAddress setAddress knownAddress items)
    (context : SearchContext env arrayAddress setAddress knownAddress items)
    (getI : env "i" = some (number i)) (getJ : env "j" = some (number j))
    (first : items[i]? = some a) (second : items[j]? = some b) (increasing : a ≤ b)
    (rejected : ProjectEulerSolutions.PrimeBucketSearch.accepted items a b = false) :
    ∃ finalEnv extra, BlockContinues program env pairBody heap finalEnv (heap ++ extra) ∧
      SearchContext finalEnv arrayAddress setAddress knownAddress items ∧ finalEnv "i" = env "i" := by
  let c := ProjectEulerSolutions.PrimeBucketSearch.third a b
  let afterPair := pairEnv env a b
  let firstHeap := heap ++ [Mutable.Object.tuple [number a, number b]]
  have constructed := pair_prefix heap env arrayAddress i j a b items context.getArray getI getJ snapshot.arrayFound first second increasing
  have locals := context.pair a b
  have extended := snapshot.append [Mutable.Object.tuple [number a, number b]]
  have memberTest := candidate_present firstHeap afterPair setAddress c items
    (by simp [afterPair, pairEnv, setLocal, c]) locals.getSet extended.setFound
  by_cases present : c ∈ items
  · let ready := setLocal afterPair "seq" (.ref firstHeap.length)
    let finalHeap := firstHeap ++ [Mutable.Object.tuple [number a, number b, number c]]
    have failed : ¬ (c ∈ items ∧ (a,b,c) ≠ ProjectEulerSolutions.PrimePermutationGroups.known) :=
      of_decide_eq_false rejected
    have same : (a,b,c) = ProjectEulerSolutions.PrimePermutationGroups.known := by
      by_contra different
      exact failed ⟨present, different⟩
    have allocated := candidate_result firstHeap afterPair a b c
      (by simp [afterPair, pairEnv, setLocal]) (by simp [afterPair, pairEnv, setLocal])
      (by simp [afterPair, pairEnv, setLocal, c])
    have compared := fresh_comparison firstHeap afterPair knownAddress a b c locals.getKnown extended.knownFound
    have excluded : BlockContinues program ready [acceptedBody[1]'(by decide)] finalHeap ready finalHeap := by
      apply BlockContinues.branch (flag := false) compared (by simp [same, Mutable.truth, scalarTruth])
      exact .nil
    have skipped : BlockContinues program afterPair (pairBody.drop 3) firstHeap ready finalHeap := by
      apply BlockContinues.branch (flag := true) memberTest (by simp [present, Mutable.truth, scalarTruth])
      exact allocated.append excluded
    refine ⟨ready, [.tuple [number a, number b], .tuple [number a, number b, number c]], ?_,
      locals.seq firstHeap.length, ?_⟩
    · simpa only [firstHeap, finalHeap, List.append_assoc, List.singleton_append] using constructed.append skipped
    · simp [ready, afterPair, pairEnv, setLocal]
  · have skipped : BlockContinues program afterPair (pairBody.drop 3) firstHeap afterPair firstHeap := by
      apply BlockContinues.branch (flag := false) memberTest (by simp [present, Mutable.truth, scalarTruth])
      exact .nil
    exact ⟨afterPair, [.tuple [number a, number b]], constructed.append skipped, locals,
      by simp [afterPair, pairEnv, setLocal]⟩

/-- A scan either continues with fresh garbage or returns a freshly allocated row. -/
private def Scan (invariant : HeapCalls.Env → Prop) (env : HeapCalls.Env) (body : List Stmt)
    (heap : Mutable.Heap) : Option ProjectEulerSolutions.PrimePermutationGroups.Row → Prop
  | none => ∃ finalEnv extra, BlockContinues program env body heap finalEnv (heap ++ extra) ∧ invariant finalEnv
  | some answer => ∀ following, ∃ address extra,
      BlockReturns program env (body ++ following) heap (.ref address) (heap ++ extra) ∧
      (heap ++ extra)[address]? = some (.tuple (rowValues answer))

theorem pair_result (heap : Mutable.Heap) (env : HeapCalls.Env) (arrayAddress setAddress knownAddress i j a b : Nat)
    (items : List Nat) (snapshot : SearchHeap heap arrayAddress setAddress knownAddress items)
    (context : SearchContext env arrayAddress setAddress knownAddress items)
    (getI : env "i" = some (number i)) (getJ : env "j" = some (number j))
    (first : items[i]? = some a) (second : items[j]? = some b) (increasing : a ≤ b) :
    Scan (fun finalEnv => SearchContext finalEnv arrayAddress setAddress knownAddress items ∧
      finalEnv "i" = some (number i)) env pairBody heap
      (if ProjectEulerSolutions.PrimeBucketSearch.accepted items a b then
        some (ProjectEulerSolutions.PrimeBucketSearch.candidate a b) else none) := by
  cases verdict : ProjectEulerSolutions.PrimeBucketSearch.accepted items a b with
  | false =>
      obtain ⟨finalEnv, extra, continued, locals, stable⟩ :=
        pair_skip heap env arrayAddress setAddress knownAddress i j a b items snapshot context getI getJ first second increasing verdict
      exact ⟨finalEnv, extra, continued, locals, stable.trans getI⟩
  | true =>
      intro following
      let middle : Mutable.Heap := heap ++ [.tuple [number a, number b]]
      refine ⟨middle.length,
        [.tuple [number a, number b], .tuple (rowValues (ProjectEulerSolutions.PrimeBucketSearch.candidate a b))], ?_, ?_⟩
      · simpa only [List.append_assoc, List.singleton_append] using
          pair_return heap env arrayAddress setAddress knownAddress i j a b items snapshot context getI getJ first second increasing verdict following
      · simp [middle]

private theorem scan_weaken (invariant weaker : HeapCalls.Env → Prop)
    {result : Option ProjectEulerSolutions.PrimePermutationGroups.Row}
    (implication : ∀ finalEnv, invariant finalEnv → weaker finalEnv)
    (executed : Scan invariant env body heap result) : Scan weaker env body heap result := by
  cases result with
  | none =>
      obtain ⟨finalEnv, extra, continued, valid⟩ := executed
      exact ⟨finalEnv, extra, continued, implication finalEnv valid⟩
  | some answer => exact executed

private theorem scan_prepend {result : Option ProjectEulerSolutions.PrimePermutationGroups.Row}
    (continued : BlockContinues program env body heap middleEnv (heap ++ extra))
    (rest : Scan invariant middleEnv following (heap ++ extra) result) :
    Scan invariant env (body ++ following) heap result := by
  cases result with
  | none =>
      obtain ⟨finalEnv, more, finished, valid⟩ := rest
      refine ⟨finalEnv, extra ++ more, ?_, valid⟩
      simpa only [List.append_assoc] using continued.append finished
  | some answer =>
      intro after
      obtain ⟨address, more, returned, found⟩ := rest after
      refine ⟨address, extra ++ more, ?_, ?_⟩
      · simpa only [List.append_assoc] using continued.returns returned
      · simpa only [List.append_assoc] using found

private theorem scan_for_range
    {result : Option ProjectEulerSolutions.PrimePermutationGroups.Row}
    (arguments : Evaluates.Arguments program env expressions heap values heap)
    (bounds : rangeBounds values = .ok (start, stop, stride))
    (executed : Scan invariant env [.rangeStepLoop name start stop stride body []] heap result) :
    Scan invariant env [.forIn (.name name) (.call (.name "range") expressions) body []] heap result := by
  cases result with
  | none =>
      obtain ⟨finalEnv, extra, continued, valid⟩ := executed
      exact ⟨finalEnv, extra, BlockContinues.for_range arguments bounds continued, valid⟩
  | some answer =>
      intro following
      obtain ⟨address, extra, returned, found⟩ := executed following
      exact ⟨address, extra, BlockReturns.for_range arguments bounds returned, found⟩

private theorem scan_range_return (active : Calls.rangeContinues next stop 1 = true)
    (executed : Scan invariant (setLocal env name (number next)) body heap (some answer)) :
    Scan invariant env [.rangeStepLoop name next stop 1 body []] heap (some answer) := by
  intro following
  obtain ⟨address, extra, returned, found⟩ :=
    executed ([Calls.loopContinuation body (.rangeStepLoop name (next + 1) stop 1 body [])] ++ following)
  refine ⟨address, extra, ?_, found⟩
  apply BlockReturns.range_step active
  simpa only [List.append_assoc] using returned

private theorem scan_range_continue (active : Calls.rangeContinues next stop 1 = true)
    {result : Option ProjectEulerSolutions.PrimePermutationGroups.Row}
    (control : Calls.loopControl body = false)
    (continued : BlockContinues program (setLocal env name (number next)) body heap middleEnv (heap ++ extra))
    (rest : Scan invariant middleEnv [.rangeStepLoop name (next + 1) stop 1 body []] (heap ++ extra) result) :
    Scan invariant env [.rangeStepLoop name next stop 1 body []] heap result := by
  cases result with
  | none =>
      obtain ⟨finalEnv, more, finished, valid⟩ := rest
      refine ⟨finalEnv, extra ++ more, ?_, valid⟩
      apply BlockContinues.range_step active
      simpa only [Calls.loopContinuation, control, Bool.false_eq_true, if_false, List.append_assoc] using
        continued.append finished
  | some answer =>
      intro following
      obtain ⟨address, more, returned, found⟩ := rest following
      refine ⟨address, extra ++ more, ?_, ?_⟩
      · apply BlockReturns.range_step active
        simpa only [Calls.loopContinuation, control, Bool.false_eq_true, if_false, List.append_assoc,
          List.singleton_append] using continued.returns returned
      · simpa only [List.append_assoc] using found

private theorem pairBody_control : Calls.loopControl pairBody = false := rfl
private theorem rowBody_control : Calls.loopControl rowBody = false := rfl

/-- The original inner range reads every remaining pair and returns exactly the native row scan. -/
theorem inner_loop (items : List Nat) (i a : Nat) (first : items[i]? = some a)
    (remaining : List Nat) (j : Nat) (suffix : items.drop j = remaining)
    (larger : ∀ b ∈ remaining, a < b)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (arrayAddress setAddress knownAddress : Nat)
    (snapshot : SearchHeap heap arrayAddress setAddress knownAddress items)
    (context : SearchContext env arrayAddress setAddress knownAddress items)
    (getI : env "i" = some (number i)) :
    Scan (fun finalEnv => SearchContext finalEnv arrayAddress setAddress knownAddress items ∧
      finalEnv "i" = some (number i)) env [.rangeStepLoop "j" j items.length 1 pairBody []] heap
      (ProjectEulerSolutions.PrimeBucketSearch.row items a remaining) := by
  induction remaining generalizing j heap env with
  | nil =>
      have empty := congrArg List.length suffix
      simp only [List.length_drop, List.length_nil] at empty
      refine ⟨env, [], ?_, context, getI⟩
      simp only [List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | cons head tail ih =>
      let bound := setLocal env "j" (number j)
      have selected : items[j]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := j)).symm
      have inside := (List.getElem?_eq_some_iff.mp selected).1
      have active : Calls.rangeContinues j items.length 1 = true := by simp [Calls.rangeContinues]; omega
      have getI' : bound "i" = some (number i) := by simpa [bound, setLocal] using getI
      have increasing : a ≤ head := Nat.le_of_lt (larger head List.mem_cons_self)
      cases verdict : ProjectEulerSolutions.PrimeBucketSearch.accepted items a head with
      | true =>
          have returned := pair_result heap bound arrayAddress setAddress knownAddress i j a head items
            snapshot (context.bind_j j) getI' (by simp [bound, setLocal]) first selected increasing
          have native : ProjectEulerSolutions.PrimeBucketSearch.row items a (head :: tail) =
              some (ProjectEulerSolutions.PrimeBucketSearch.candidate a head) := by
            simp [ProjectEulerSolutions.PrimeBucketSearch.row, verdict]
          rw [native]
          apply scan_range_return active
          simpa only [verdict, if_pos (rfl : (true : Bool) = true)] using returned
      | false =>
          obtain ⟨nextEnv, extra, continued, locals, stable⟩ := pair_skip heap bound
            arrayAddress setAddress knownAddress i j a head items snapshot (context.bind_j j)
            getI' (by simp [bound, setLocal]) first selected increasing verdict
          have next := ih (j := j + 1) (suffix := by simp [List.drop_add_one_eq_tail_drop, suffix])
            (larger := fun b member => larger b (List.mem_cons_of_mem _ member)) (heap := heap ++ extra)
            (env := nextEnv) (snapshot.append extra) locals (stable.trans getI')
          have native : ProjectEulerSolutions.PrimeBucketSearch.row items a (head :: tail) =
              ProjectEulerSolutions.PrimeBucketSearch.row items a tail := by
            simp [ProjectEulerSolutions.PrimeBucketSearch.row, verdict]
          rw [native]
          apply scan_range_continue active pairBody_control continued
          simpa only [Int.natCast_add, Int.natCast_one] using next

/-- The original inner for statement evaluates its captured range bounds before scanning. -/
theorem row_result (items : List Nat) (i a : Nat) (first : items[i]? = some a)
    (remaining : List Nat) (suffix : items.drop (i + 1) = remaining) (larger : ∀ b ∈ remaining, a < b)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (arrayAddress setAddress knownAddress : Nat)
    (snapshot : SearchHeap heap arrayAddress setAddress knownAddress items)
    (context : SearchContext env arrayAddress setAddress knownAddress items)
    (getI : env "i" = some (number i)) :
    Scan (fun finalEnv => SearchContext finalEnv arrayAddress setAddress knownAddress items)
      env rowBody heap (ProjectEulerSolutions.PrimeBucketSearch.row items a remaining) := by
  have scanned := inner_loop items i a first remaining (i + 1) suffix larger heap env
    arrayAddress setAddress knownAddress snapshot context getI
  have arguments : Evaluates.Arguments program env
      [.binOp (.name "i") .add (.constant (.int 1)), .name "m"] heap
      [number (i + 1 : Nat), number items.length] heap := by
    refine .cons ?_ (.cons (Evaluates.lookup context.getLength) .nil)
    exact Evaluates.binary (av := number i) (bv := number 1) (middle := heap) (afterRight := heap)
      (Evaluates.lookup getI) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
  have entered := scan_for_range arguments (start := (i + 1 : Nat)) (stop := items.length) (stride := 1) rfl scanned
  exact scan_weaken _ _ (fun _ valid => valid.1) entered

/-- The original outer range returns exactly the ordered pair search, including an exhausted final row. -/
theorem outer_loop (items : List Nat) (ordered : items.Pairwise (· < ·))
    (remaining : List Nat) (i : Nat) (suffix : items.drop i = remaining)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (arrayAddress setAddress knownAddress : Nat)
    (snapshot : SearchHeap heap arrayAddress setAddress knownAddress items)
    (context : SearchContext env arrayAddress setAddress knownAddress items) :
    Scan (fun finalEnv => SearchContext finalEnv arrayAddress setAddress knownAddress items) env
      [.rangeStepLoop "i" i items.length 1 rowBody []] heap
      (ProjectEulerSolutions.PrimeBucketSearch.pairs items remaining) := by
  induction remaining generalizing i heap env with
  | nil =>
      have empty := congrArg List.length suffix
      simp only [List.length_drop, List.length_nil] at empty
      refine ⟨env, [], ?_, context⟩
      simp only [List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | cons head tail ih =>
      let bound := setLocal env "i" (number i)
      have selected : items[i]? = some head := by
        simpa [suffix] using (List.head?_drop (l := items) (i := i)).symm
      have inside := (List.getElem?_eq_some_iff.mp selected).1
      have active : Calls.rangeContinues i items.length 1 = true := by simp [Calls.rangeContinues]; omega
      have remainingOrdered : (head :: tail).Pairwise (· < ·) := by
        rw [← suffix]
        exact ordered.drop
      have rowScan := row_result items i head selected tail
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) (List.pairwise_cons.mp remainingOrdered).1
        heap bound arrayAddress setAddress knownAddress snapshot (context.bind_i i) (by simp [bound, setLocal])
      cases verdict : ProjectEulerSolutions.PrimeBucketSearch.row items head tail with
      | some answer =>
          have native : ProjectEulerSolutions.PrimeBucketSearch.pairs items (head :: tail) = some answer := by
            simp [ProjectEulerSolutions.PrimeBucketSearch.pairs, verdict]
          rw [native]
          apply scan_range_return active
          simpa only [verdict] using rowScan
      | none =>
          have exhausted : Scan (fun finalEnv => SearchContext finalEnv arrayAddress setAddress knownAddress items)
              bound rowBody heap none := by simpa only [verdict] using rowScan
          obtain ⟨nextEnv, extra, continued, locals⟩ := exhausted
          have next := ih (i := i + 1) (suffix := by simp [List.drop_add_one_eq_tail_drop, suffix])
            (heap := heap ++ extra) (env := nextEnv) (snapshot.append extra) locals
          have native : ProjectEulerSolutions.PrimeBucketSearch.pairs items (head :: tail) =
              ProjectEulerSolutions.PrimeBucketSearch.pairs items tail := by
            simp [ProjectEulerSolutions.PrimeBucketSearch.pairs, verdict]
          rw [native]
          apply scan_range_continue active rowBody_control continued
          simpa only [Int.natCast_add, Int.natCast_one] using next

theorem bucket_scan (items : List Nat) (ordered : items.Pairwise (· < ·))
    (heap : Mutable.Heap) (env : HeapCalls.Env) (arrayAddress setAddress knownAddress : Nat)
    (snapshot : SearchHeap heap arrayAddress setAddress knownAddress items)
    (context : SearchContext env arrayAddress setAddress knownAddress items) :
    Scan (fun finalEnv => SearchContext finalEnv arrayAddress setAddress knownAddress items)
      env [searchBody[4]'(by decide)] heap (ProjectEulerSolutions.PrimeBucketSearch.pairs items items) := by
  have scanned := outer_loop items ordered items 0 rfl heap env arrayAddress setAddress knownAddress snapshot context
  exact scan_for_range (.cons (Evaluates.lookup context.getLength) .nil)
    (values := [number items.length]) (start := 0) (stop := items.length) (stride := 1) rfl scanned

/-- A complete nontrivial bucket iteration agrees with the finite pair search. -/
theorem bucket_result (items : List Nat) (ordered : items.Pairwise (· < ·)) (large : 3 ≤ items.length)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (arrayAddress knownAddress : Nat)
    (getArray : env "arr" = some (.ref arrayAddress)) (getKnown : env "target_known" = some (.ref knownAddress))
    (arrayFound : heap[arrayAddress]? = some (.list (numbers items)))
    (knownFound : heap[knownAddress]? = some (.tuple knownValues)) :
    Scan (fun finalEnv => finalEnv "target_known" = some (.ref knownAddress)) env searchBody heap
      (ProjectEulerSolutions.PrimeBucketSearch.pairs items items) := by
  let ready := setLocal (setLocal env "s" (.ref heap.length)) "m" (number items.length)
  let afterSet := heap ++ [Mutable.Object.set (ProjectEulerSolutions.PrimeBucketSearch.setKeys items)]
  have snapshot : SearchHeap afterSet arrayAddress heap.length knownAddress items :=
    ⟨prefix_found heap _ _ _ arrayFound, by simp [afterSet], prefix_found heap _ _ _ knownFound⟩
  have context : SearchContext ready arrayAddress heap.length knownAddress items :=
    ⟨by simpa [ready, setLocal] using getArray, by simp [ready, setLocal], by simp [ready, setLocal],
      by simpa [ready, setLocal] using getKnown⟩
  have scanned := bucket_scan items ordered afterSet ready arrayAddress heap.length knownAddress snapshot context
  have weaker := scan_weaken
    (fun finalEnv => SearchContext finalEnv arrayAddress heap.length knownAddress items)
    (fun finalEnv => finalEnv "target_known" = some (.ref knownAddress)) (fun _ valid => valid.getKnown) scanned
  exact scan_prepend (bucket_setup heap env arrayAddress items getArray arrayFound ordered large) weaker

private def itemValues (bucket : ProjectEulerSolutions.StringBuckets.Bucket) : List Mutable.Value :=
  [.scalar (.string bucket.key), .ref bucket.address]
private def itemEnv (env : HeapCalls.Env) (bucket : ProjectEulerSolutions.StringBuckets.Bucket)
    (address : Nat) : HeapCalls.Env :=
  assignNames (setLocal env itemName (.ref address)) ["key", "arr"] (itemValues bucket)

private theorem itemBody_loop (next : Stmt) : Calls.loopContinuation itemBody next = .loopResume next := rfl

private theorem item_array (env : HeapCalls.Env) (bucket : ProjectEulerSolutions.StringBuckets.Bucket)
    (address : Nat) : itemEnv env bucket address "arr" = some (.ref bucket.address) := by
  simp [itemEnv, itemValues, assignNames, setLocal]

private theorem item_known (getKnown : env "target_known" = some (.ref knownAddress))
    (bucket : ProjectEulerSolutions.StringBuckets.Bucket) (address : Nat) :
    itemEnv env bucket address "target_known" = some (.ref knownAddress) := by
  simpa [itemEnv, itemValues, assignNames, setLocal, itemName, itemTarget, comprehensionTargetName] using getKnown

private theorem unpack_item (heap : Mutable.Heap) (env : HeapCalls.Env)
    (bucket : ProjectEulerSolutions.StringBuckets.Bucket) (address : Nat)
    (found : heap[address]? = some (.tuple (itemValues bucket)))
    (rest : BlockReturns program (itemEnv env bucket address) (searchBody ++ following) heap answer finalHeap) :
    BlockReturns program (setLocal env itemName (.ref address)) (itemBody ++ following) heap answer finalHeap := by
  apply BlockReturns.unpack (names := ["key", "arr"]) (items := itemValues bucket)
    (value := .ref address) (middle := heap) (afterUnpack := heap) rfl (Evaluates.lookup rfl)
    (by simp [Mutable.sequence, found]) rfl rest

private theorem item_step_return (heap : Mutable.Heap) (env : HeapCalls.Env)
    (dictionary position : Nat) (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (bucket : ProjectEulerSolutions.StringBuckets.Bucket)
    (fetched : dictionaryItemsNext dictionary (itemKeys buckets) position heap =
      some (.ok (some (.scalar (.string bucket.key), .ref bucket.address)), heap))
    (rest : BlockReturns program (itemEnv env bucket heap.length)
      (searchBody ++ [.loopResume (itemLoop dictionary buckets (position + 1))] ++ following)
      (heap ++ [.tuple (itemValues bucket)]) answer finalHeap) :
    BlockReturns program env (itemLoop dictionary buckets position :: following) heap answer finalHeap := by
  apply BlockReturns.dictionary_items_step (pair := .ref heap.length)
    (afterTuple := heap ++ [.tuple (itemValues bucket)]) fetched
    (by simp [Mutable.allocateTuple, itemValues, Except.map])
  apply unpack_item _ env bucket heap.length (by simp)
  simpa only [itemBody_loop, List.append_assoc] using rest

private theorem winner_tail (head : ProjectEulerSolutions.StringBuckets.Bucket)
    (tail : List ProjectEulerSolutions.StringBuckets.Bucket)
    (success : ∃ bucket ∈ head :: tail, ProjectEulerSolutions.PrimeBucketSearch.pairs bucket.values bucket.values =
      some ProjectEulerSolutions.PrimePermutationGroups.winner)
    (exhausted : ProjectEulerSolutions.PrimeBucketSearch.pairs head.values head.values = none) :
    ∃ bucket ∈ tail, ProjectEulerSolutions.PrimeBucketSearch.pairs bucket.values bucket.values =
      some ProjectEulerSolutions.PrimePermutationGroups.winner := by
  obtain ⟨bucket, member, result⟩ := success
  rcases List.mem_cons.mp member with rfl | member
  · rw [exhausted] at result
    cases result
  · exact ⟨bucket, member, result⟩

/-- The live dictionary scan reaches a winning bucket and preserves every preceding heap object. -/
theorem items_loop (buckets : List ProjectEulerSolutions.StringBuckets.Bucket)
    (partition : ProjectEulerSolutions.StringBuckets.Partition keyOf primes4d buckets)
    (remaining : List ProjectEulerSolutions.StringBuckets.Bucket) (position : Nat)
    (suffix : buckets.drop position = remaining)
    (success : ∃ bucket ∈ remaining, ProjectEulerSolutions.PrimeBucketSearch.pairs bucket.values bucket.values =
      some ProjectEulerSolutions.PrimePermutationGroups.winner)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (dictionary knownAddress : Nat)
    (view : ProjectEulerSolutions.StringBuckets.View heap dictionary buckets)
    (knownFound : heap[knownAddress]? = some (.tuple knownValues))
    (getKnown : env "target_known" = some (.ref knownAddress)) :
    Scan (fun _ => True) env [itemLoop dictionary buckets position] heap
      (some ProjectEulerSolutions.PrimePermutationGroups.winner) := by
  induction remaining generalizing position heap env with
  | nil => simp at success
  | cons head tail ih =>
      let nextHeap := heap ++ [Mutable.Object.tuple (itemValues head)]
      let unpacked := itemEnv env head heap.length
      have member : head ∈ buckets :=
        List.mem_of_mem_drop (show head ∈ buckets.drop position by rw [suffix]; simp)
      have selected : buckets[position]? = some head := by
        simpa [suffix] using (List.head?_drop (l := buckets) (i := position)).symm
      have fetched : dictionaryItemsNext dictionary (itemKeys buckets) position heap =
          some (.ok (some (.scalar (.string head.key), .ref head.address)), heap) := by
        simpa only [itemKeys, ProjectEulerSolutions.StringBuckets.entries, List.getElem?_map, selected,
          Option.map_some, ProjectEulerSolutions.StringBuckets.entry] using
          dictionaryItemsNext_dict (heap := heap) (address := dictionary) (position := position)
            (entries := ProjectEulerSolutions.StringBuckets.entries buckets) view.found
      have extended := view.append [Mutable.Object.tuple (itemValues head)]
      have arrayAfter : nextHeap[head.address]? = some (.list (numbers head.values)) := extended.lists head member
      have knownAfter : nextHeap[knownAddress]? = some (.tuple knownValues) := prefix_found heap _ _ _ knownFound
      have getKnown' : unpacked "target_known" = some (.ref knownAddress) := item_known getKnown head heap.length
      have ordered := bucket_ordered buckets partition head member
      cases verdict : ProjectEulerSolutions.PrimeBucketSearch.pairs head.values head.values with
      | some answer =>
          have same := bucket_search_sound buckets partition head member answer verdict
          have large : 3 ≤ head.values.length := by
            by_contra small
            have empty := ProjectEulerSolutions.PrimeBucketSearch.short_none head.values ordered (by omega)
            rw [empty] at verdict
            cases verdict
          have scanned := bucket_result head.values ordered large nextHeap unpacked head.address knownAddress
            (item_array env head heap.length) getKnown' arrayAfter knownAfter
          have returned : Scan (fun finalEnv => finalEnv "target_known" = some (.ref knownAddress))
              unpacked searchBody nextHeap (some ProjectEulerSolutions.PrimePermutationGroups.winner) := by
            simpa only [verdict, same] using scanned
          intro following
          obtain ⟨address, more, finished, found⟩ :=
            returned ([.loopResume (itemLoop dictionary buckets (position + 1))] ++ following)
          refine ⟨address, [.tuple (itemValues head)] ++ more, ?_, ?_⟩
          · have complete := item_step_return heap env dictionary position buckets head fetched
              (by simpa only [List.append_assoc] using finished)
            simpa only [nextHeap, List.append_assoc] using complete
          · simpa only [nextHeap, List.append_assoc] using found
      | none =>
          have tailSuccess := winner_tail head tail success verdict
          by_cases short : head.values.length < 3
          · have next := ih (position := position + 1)
              (suffix := by simp [List.drop_add_one_eq_tail_drop, suffix]) (success := tailSuccess)
              (heap := nextHeap) (env := unpacked) extended knownAfter getKnown'
            intro following
            obtain ⟨address, more, finished, found⟩ := next following
            have size : Evaluates program unpacked (.builtinCall .len [.name "arr"]) nextHeap
                (number head.values.length) nextHeap := by
              apply Evaluates.builtin (values := [.ref head.address])
                (.cons (Evaluates.lookup (item_array env head heap.length)) .nil)
              simp [primitive, Mutable.length, arrayAfter, numbers]
            have skipped : BlockReturns program unpacked
                (searchBody ++ [.loopResume (itemLoop dictionary buckets (position + 1))] ++ following)
                nextHeap (.ref address) (nextHeap ++ more) := by
              apply BlockReturns.branch (flag := true) (value := flag true) (middle := nextHeap)
              · apply Evaluates.compare (av := number head.values.length) (bv := number 3)
                  (middle := nextHeap) (afterRight := nextHeap) size Evaluates.constant
                simp [HeapCalls.compare, scalarCompare, evalCmpOp, show (head.values.length : Int) < 3 by omega]
              · rfl
              · exact BlockReturns.continue_loop (next := itemLoop dictionary buckets (position + 1))
                  (following := following) rfl finished
            refine ⟨address, [.tuple (itemValues head)] ++ more, ?_, ?_⟩
            · have complete := item_step_return heap env dictionary position buckets head fetched skipped
              simpa only [nextHeap, List.append_assoc] using complete
            · simpa only [nextHeap, List.append_assoc] using found
          · have scanned := bucket_result head.values ordered (by omega) nextHeap unpacked head.address knownAddress
              (item_array env head heap.length) getKnown' arrayAfter knownAfter
            have exhausted : Scan (fun finalEnv => finalEnv "target_known" = some (.ref knownAddress))
                unpacked searchBody nextHeap none := by simpa only [verdict] using scanned
            obtain ⟨nextEnv, extra, continued, known⟩ := exhausted
            have next := ih (position := position + 1)
              (suffix := by simp [List.drop_add_one_eq_tail_drop, suffix]) (success := tailSuccess)
              (heap := nextHeap ++ extra) (env := nextEnv) (extended.append extra)
              (prefix_found nextHeap extra _ _ knownAfter) known
            intro following
            obtain ⟨address, more, finished, found⟩ := next following
            have resumed := BlockReturns.loop_resume finished
            have searched : BlockReturns program unpacked
                (searchBody ++ [.loopResume (itemLoop dictionary buckets (position + 1))] ++ following)
                nextHeap (.ref address) ((nextHeap ++ extra) ++ more) := by
              simpa only [List.append_assoc, List.singleton_append] using continued.returns resumed
            refine ⟨address, [.tuple (itemValues head)] ++ (extra ++ more), ?_, ?_⟩
            · have complete := item_step_return heap env dictionary position buckets head fetched searched
              simpa only [nextHeap, List.append_assoc] using complete
            · simpa only [nextHeap, List.append_assoc] using found

/-- The complete original dictionary-based search returns the required progression. -/
theorem result (heap : Mutable.Heap) :
    ∃ address extra, ReturnsValue program "find_prime_perm_arith_sequence" [] heap (.ref address)
      (preparedHeap heap ++ extra) ∧
      (preparedHeap heap ++ extra)[address]? = some (.tuple (rowValues ProjectEulerSolutions.PrimePermutationGroups.winner)) := by
  let definition := (findFunctionIn program.body "find_prime_perm_arith_sequence").getD ([], [])
  let initial := functionEnv definition.1 definition.2 [] []
  obtain ⟨ready, extra, buckets, knownAddress, prepared, view, partition, getGroups, getKnown, knownFound⟩ :=
    search_ready_result heap initial
  have scanned := items_loop buckets partition buckets 0 rfl (winner_search buckets partition)
    (preparedHeap heap ++ extra) ready (preparedHeap heap).length knownAddress view knownFound getKnown
  obtain ⟨address, more, returned, found⟩ := scanned [code[6]'(by decide)]
  have rest : BlockReturns program ready (code.drop 5) (preparedHeap heap ++ extra) (.ref address)
      ((preparedHeap heap ++ extra) ++ more) := by
    apply BlockReturns.for_dictionary_items (address := (preparedHeap heap).length)
      (object := ProjectEulerSolutions.StringBuckets.object buckets)
      (entries := (ProjectEulerSolutions.StringBuckets.entries buckets).map (fun entry => (.scalar entry.1, entry.2)))
      (middle := preparedHeap heap ++ extra) (Evaluates.lookup getGroups) view.found rfl
    simpa only [itemLoop, itemName, itemBody, itemTarget, itemKeys, List.map_map, Function.comp_def, sequenceSource] using returned
  refine ⟨address, extra ++ more, ?_, ?_⟩
  · apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    simpa only [List.take_append_drop, List.append_assoc] using prepared.returns rest
  · simpa only [List.append_assoc] using found

theorem «Returns» : (SetCalls.call program "find_prime_perm_arith_sequence" [] []).Returns := by
  obtain ⟨address, extra, returned, found⟩ := result []
  exact SetCalls.returns_of_heap returned.returns

theorem «Supported» : (SetCalls.call program "find_prime_perm_arith_sequence" [] []).Supported := by
  intro choices
  exact («Returns» choices).supported

end find_prime_perm_arith_sequence

namespace solve

private theorem winner_text :
    (("" ++ toString (2969 : Int)) ++ toString (6299 : Int)) ++ toString (9629 : Int) =
      toString (296962999629 : Int) := by
  decide +kernel

private theorem format_result (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getA : env "a" = some (number 2969)) (getB : env "b" = some (number 6299))
    (getC : env "c" = some (number 9629)) :
    Evaluates program env
      (.builtinCall .int [.binOp (.binOp (.binOp (.constant (.string "")) .add
        (.builtinCall .str [.name "a"])) .add (.builtinCall .str [.name "b"])) .add
        (.builtinCall .str [.name "c"])]) heap (number 296962999629) heap := by
  have aText : Evaluates program env (.builtinCall .str [.name "a"]) heap
      (.scalar (.string (toString (2969 : Int)))) heap :=
    Evaluates.builtin (.cons (Evaluates.lookup getA) .nil) rfl
  have bText : Evaluates program env (.builtinCall .str [.name "b"]) heap
      (.scalar (.string (toString (6299 : Int)))) heap :=
    Evaluates.builtin (.cons (Evaluates.lookup getB) .nil) rfl
  have cText : Evaluates program env (.builtinCall .str [.name "c"]) heap
      (.scalar (.string (toString (9629 : Int)))) heap :=
    Evaluates.builtin (.cons (Evaluates.lookup getC) .nil) rfl
  have joined : Evaluates program env
      (.binOp (.binOp (.binOp (.constant (.string "")) .add (.builtinCall .str [.name "a"])) .add
        (.builtinCall .str [.name "b"])) .add (.builtinCall .str [.name "c"])) heap
      (.scalar (.string (toString (296962999629 : Int)))) heap := by
    rw [← winner_text]
    exact Evaluates.binary (Evaluates.binary (Evaluates.binary Evaluates.constant aText rfl) bText rfl) cText rfl
  apply Evaluates.builtin (.cons joined .nil)
  simp only [primitive, List.mapM_cons, List.mapM_nil, pure_bind]
  rw [Calls.integerConversion_int]
  rfl

/-- The complete original solver unpacks the winning tuple and concatenates its decimal values. -/
theorem result (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "solve" [] heap (number 296962999629)
      (find_prime_perm_arith_sequence.preparedHeap heap ++ extra) := by
  obtain ⟨address, extra, searched, found⟩ := find_prime_perm_arith_sequence.result heap
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [] []
  let ready := assignNames initial ["a", "b", "c"] [number 2969, number 6299, number 9629]
  refine ⟨extra, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.unpack (names := ["a", "b", "c"]) (items := [number 2969, number 6299, number 9629])
    (middle := find_prime_perm_arith_sequence.preparedHeap heap ++ extra)
    (afterUnpack := find_prime_perm_arith_sequence.preparedHeap heap ++ extra) rfl
    (Evaluates.call .nil searched)
    (by simp [Mutable.sequence, found, find_prime_perm_arith_sequence.rowValues,
      ProjectEulerSolutions.PrimePermutationGroups.winner]) rfl
  exact BlockReturns.return_value (format_result _ ready (by simp [ready, assignNames, setLocal])
    (by simp [ready, assignNames, setLocal]) (by simp [ready, assignNames, setLocal]))

theorem «Equiv(naive_default)» :
    (SetCalls.call program "solve" [] []).Equiv
      (SetCalls.call P49Ast.module "naive_default" [] []) := by
  intro leftChoices rightChoices
  obtain ⟨extra, returned⟩ := result []
  obtain ⟨finalHeap, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p49.naive_default.result []
  apply Mutable.Call.equiv_of_finishes
    (a := (SetCalls.call program "solve" [] []).run leftChoices)
    (b := (SetCalls.call P49Ast.module "naive_default" [] []).run rightChoices)
    (SetCalls.finishes_of_heap returned.finishes leftChoices 0)
    (SetCalls.finishes_of_heap reference.finishes rightChoices 0)
  exact Mutable.Completion.equivalent_allocations []
    (find_prime_perm_arith_sequence.preparedHeap [] ++ extra) finalHeap (number 296962999629) []
    (by simp [Mutable.Heap.Valid]) trivial (by simp)

theorem «Supported» :
    (SetCalls.call program "solve" [] []).Supported ∧
      (SetCalls.call P49Ast.module "naive_default" [] []).Supported := by
  obtain ⟨extra, returned⟩ := result []
  obtain ⟨finalHeap, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p49.naive_default.result []
  exact ⟨fun choices => Execution.supported_of_finishes (SetCalls.finishes_of_heap returned.finishes choices 0),
    fun choices => Execution.supported_of_finishes (SetCalls.finishes_of_heap reference.finishes choices 0)⟩

end solve
end solvers.«49»

