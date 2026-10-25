import «.formalsw».solvers.«49.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.BooleanPrimeSieve

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
end solvers.«49»

