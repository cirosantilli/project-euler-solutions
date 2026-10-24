import «.formalsw».solvers.«46.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p46.py»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.PrimeTwiceSquareSearch
import ProjectEulerSolutions.Lib.HeapCollection

set_option Elab.async false

namespace solvers.«46»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions
private abbrev program := Solvers.P46Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev natural (n : Nat) : Mutable.Value := number n
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private def numbers (items : List Nat) : List Mutable.Value := items.map natural

namespace is_prime_with_list
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "is_prime_with_list").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[1]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []

private theorem loop (n : Nat) (items remaining : List Nat) (position address : Nat)
    (positive : ∀ p ∈ remaining, 0<p) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (sequence : Mutable.sequence (.ref address) heap=some (.ok (numbers items), heap))
    (suffix : items.drop position=remaining) (getN : env "n"=some (natural n)) :
    BlockReturns program env
      [.sequenceStepLoop "p" (.ref address) position body [], .return (.constant (.bool true))]
      heap (flag (PrimePrefix.trial n remaining)) heap := by
  induction remaining generalizing position env with
  | nil =>
      apply BlockReturns.sequence_done (source := .ref address) sequence
      · have missing : items[position]?=none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [numbers, missing]
      · exact BlockReturns.return_value Evaluates.constant
  | cons p rest ih =>
      have item : items[position]?=some p := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      let bound := setLocal env "p" (natural p)
      have getP : bound "p"=some (natural p) := rfl
      have boundN : bound "n"=some (natural n) := by simpa [bound, setLocal] using getN
      have compared : Evaluates program bound
          (.compare (.binOp (.name "p") .mult (.name "p")) [(.gt, .name "n")])
          heap (flag (decide (n<p*p))) heap := by
        apply Evaluates.compare (av := natural (p*p)) (bv := natural n) (middle := heap) (afterRight := heap)
        · exact Evaluates.binary (av := natural p) (bv := natural p)
            (Evaluates.lookup getP) (Evaluates.lookup getP) (by simp [Mutable.binary, scalarBinary])
        · exact Evaluates.lookup boundN
        · change some (Except.ok (ε := ExceptionObject) (flag (decide ((n : Int)<((p*p : Nat) : Int)))), heap)=_
          simp only [Int.ofNat_lt]
      rw [PrimePrefix.trial]
      apply BlockReturns.sequence_step (source := .ref address) (value := natural p) sequence
        (by simp [numbers, item])
      by_cases beyond : n<p*p
      · rw [if_pos beyond]
        apply BlockReturns.branch (value := flag true) (flag := true)
          (by simpa only [beyond, decide_true] using compared) rfl
        apply BlockReturns.break_loop rfl
        exact BlockReturns.return_value Evaluates.constant
      · rw [if_neg beyond]
        apply BlockReturns.branch (value := flag false) (flag := false)
          (by simpa only [beyond, decide_false] using compared) rfl
        have dividesTest : Evaluates program bound
            (.compare (.binOp (.name "n") .mod (.name "p")) [(.eq, .constant (.int 0))])
            heap (flag (decide (n%p=0))) heap := by
          apply Evaluates.compare (av := natural (n%p)) (bv := natural 0) (middle := heap) (afterRight := heap)
          · exact Evaluates.binary (av := natural n) (bv := natural p)
              (Evaluates.lookup boundN) (Evaluates.lookup getP)
              (by simp [Mutable.binary, scalarBinary, show 0<p from positive p (by simp),
                show p≠0 by have := positive p (by simp); omega])
          · exact Evaluates.constant
          · change some (Except.ok (ε := ExceptionObject) (flag (decide (((n%p : Nat) : Int)=0))), heap)=_
            simp only [Int.natCast_eq_zero]
        by_cases divides : n%p=0
        · rw [if_pos divides]
          apply BlockReturns.branch (value := flag true) (flag := true)
            (by simpa only [divides, decide_true] using dividesTest) rfl
          exact BlockReturns.return_value Evaluates.constant
        · rw [if_neg divides]
          apply BlockReturns.branch (value := flag false) (flag := false)
            (by simpa only [divides, decide_false] using dividesTest) rfl
          apply BlockReturns.loop_resume
          apply ih (position+1) (fun q member => positive q (by simp [member])) bound
          · simp [List.drop_add_one_eq_tail_drop, suffix]
          · exact boundN

/-- Trial division reads the shared list without changing its identity or contents. -/
theorem result (n : Nat) (large : 2≤n) (items : List Nat)
    (positive : ∀ p ∈ items, 0<p) (heap : Mutable.Heap) (address : Nat)
    (sequence : Mutable.sequence (.ref address) heap=some (.ok (numbers items), heap)) :
    ReturnsValue program "is_prime_with_list" [natural n, .ref address] heap
      (flag (PrimePrefix.trial n items)) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (value := flag false) (flag := false)
    (Evaluates.compare (av := natural n) (bv := natural 2) (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup rfl) Evaluates.constant
      (by
        change some (Except.ok (ε := ExceptionObject) (flag (decide ((n : Int)<2))), heap)=_
        simp only [show ¬ (n : Int)<2 by omega, decide_false])) rfl
  apply BlockReturns.for_sequence (by intro expressions h; cases h)
    (Evaluates.lookup rfl) sequence
  exact loop n items items 0 address positive heap _ sequence (by simp) rfl
end is_prime_with_list

private theorem last_result (items : List Nat) (top : Nat) (last : items.getLast?=some top)
    (heap : Mutable.Heap) (address : Nat) (found : heap[address]?=some (.list (numbers items))) :
    Mutable.getItem (.ref address) (number (-1)) heap=some (.ok (natural top), heap) := by
  obtain ⟨before, rfl⟩ := List.getLast?_eq_some_iff.mp last
  simp [Mutable.getItem, found, Mutable.index, numbers, Except.map]

private theorem get_last (top : Nat) (prime : Nat.Prime top) (base : Mutable.Heap)
    (env : HeapCalls.Env) (getPrimes : env "primes"=some (.ref base.length)) :
    Evaluates program env (.subscript (.name "primes") (.unaryOp .uSub (.constant (.int 1))))
      (HeapCollection.working base (numbers (PrimePrefix.through top)) []) (natural top)
      (HeapCollection.working base (numbers (PrimePrefix.through top)) []) := by
  apply Evaluates.subscript (by intro a b c h; cases h) (Evaluates.lookup getPrimes) (Evaluates.unary Evaluates.constant rfl)
  exact last_result _ top (PrimePrefix.last_through top prime) _ base.length (HeapCollection.found _ _ _)

private abbrev primeHeap (base : Mutable.Heap) (top : Nat) : Mutable.Heap :=
  HeapCollection.working base (numbers (PrimePrefix.through top)) []

private theorem implicit_none (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockReturns program env [] heap (.scalar .none) heap :=
  fun _ _ => .next rfl .refl

namespace ensure_primes_up_to
private def condition : Expr := .compare
  (.subscript (.name "primes") (.unaryOp .uSub (.constant (.int 1)))) [(.lt, .name "n")]
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "ensure_primes_up_to").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[2]? : Option Stmt) with | some (.whileDo _ body _) => body | _ => []

private theorem guard (n top : Nat) (prime : Nat.Prime top) (base : Mutable.Heap)
    (env : HeapCalls.Env) (getN : env "n"=some (natural n))
    (getPrimes : env "primes"=some (.ref base.length)) :
    Evaluates program env condition (primeHeap base top) (flag (decide (top<n))) (primeHeap base top) := by
  apply Evaluates.compare (av := natural top) (bv := natural n) (middle := primeHeap base top)
    (afterRight := primeHeap base top) (get_last top prime base env getPrimes) (Evaluates.lookup getN)
  change some (Except.ok (ε := ExceptionObject) (flag (decide ((top : Int)<(n : Int)))), primeHeap base top)=_
  simp only [Int.ofNat_lt]

private theorem stopped (n top : Nat) (prime : Nat.Prime top) (reached : n≤top)
    (base : Mutable.Heap) (env : HeapCalls.Env) (getN : env "n"=some (natural n))
    (getPrimes : env "primes"=some (.ref base.length)) :
    BlockReturns program env [.whileDo condition body []] (primeHeap base top) (.scalar .none) (primeHeap base top) := by
  apply BlockReturns.while_loop (flag := false)
    (by simpa only [show ¬ top<n by omega, decide_false] using guard n top prime base env getN getPrimes) rfl
  exact implicit_none env (primeHeap base top)

private theorem loop (n q : Nat) (qPrime : Nat.Prime q) (covered : n≤q)
    (fuel candidate top : Nat) (distance : candidate+2*fuel=q)
    (prime : Nat.Prime top) (large : 3≤top) (odd : candidate%2=1)
    (complete : PrimePrefix.through top=PrimePrefix.below candidate)
    (base : Mutable.Heap) (env : HeapCalls.Env) (getN : env "n"=some (natural n))
    (getPrimes : env "primes"=some (.ref base.length)) (getCandidate : env "candidate"=some (natural candidate)) :
    ∃ finalTop, Nat.Prime finalTop ∧ 3≤finalTop ∧ n≤finalTop ∧
      BlockReturns program env [.whileDo condition body []]
        (primeHeap base top) (.scalar .none) (primeHeap base finalTop) := by
  induction fuel generalizing candidate top env with
  | zero =>
      by_cases reached : n≤top
      · exact ⟨top, prime, large, reached, stopped n top prime reached base env getN getPrimes⟩
      have candidateQ : candidate=q := by omega
      have candidatePrime : Nat.Prime candidate := candidateQ ▸ qPrime
      have topBefore : top<candidate := by
        have member : top ∈ PrimePrefix.through top := (PrimePrefix.mem_through top top).mpr ⟨by omega, prime⟩
        rw [complete] at member
        exact (PrimePrefix.mem_below top candidate).mp member |>.1
      have targetReached : n≤candidate := by omega
      have nextComplete : PrimePrefix.through candidate=PrimePrefix.below candidate++[candidate] := by
        simp [PrimePrefix.through, PrimePrefix.below_succ, candidatePrime]
      have trialResult : Evaluates program env (.call (.name "is_prime_with_list") [.name "candidate", .name "primes"])
          (primeHeap base top) (flag true) (primeHeap base top) := by
        apply Evaluates.call (.cons (Evaluates.lookup getCandidate) (.cons (Evaluates.lookup getPrimes) .nil))
        have run := is_prime_with_list.result candidate (by omega) (PrimePrefix.through top)
          (by intro p member; have := (PrimePrefix.mem_through p top).mp member |>.2.two_le; omega)
          (primeHeap base top) base.length (by simp [Mutable.sequence, primeHeap, HeapCollection.found])
        simpa only [complete, PrimePrefix.trial_below candidate (by omega), candidatePrime, decide_true] using run
      refine ⟨candidate, candidatePrime, by omega, targetReached, ?_⟩
      apply BlockReturns.while_loop (flag := true)
        (by simpa only [show top<n by omega, decide_true] using guard n top prime base env getN getPrimes) rfl
      apply BlockReturns.branch (flag := true) trialResult rfl
      apply BlockReturns.discard (value := .scalar .none) (middle := primeHeap base candidate)
      · have appendResult := Evaluates.append (module := program) (globals := []) (Evaluates.lookup getPrimes)
          (HeapCollection.found base (numbers (PrimePrefix.through top)) [])
          (Evaluates.lookup getCandidate) (HeapCollection.found base (numbers (PrimePrefix.through top)) [])
        simpa only [primeHeap, HeapCollection.store, nextComplete, complete, numbers, List.map_append,
          List.map_cons, List.map_nil] using appendResult
      · apply BlockReturns.augAssign (value := natural (candidate+2))
          (middle := primeHeap base candidate) (afterOperation := primeHeap base candidate) getCandidate Evaluates.constant
        · simp [inplace, Mutable.binary, scalarBinary]
        · have next := stopped n candidate candidatePrime targetReached base
            (setLocal env "candidate" (natural (candidate+2))) (by simpa [setLocal] using getN)
            (by simpa [setLocal] using getPrimes)
          simpa only [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            Bool.false_or, Bool.false_eq_true, if_false, List.nil_append] using next
  | succ fuel ih =>
      by_cases reached : n≤top
      · exact ⟨top, prime, large, reached, stopped n top prime reached base env getN getPrimes⟩
      have topBefore : top<candidate := by
        have member : top ∈ PrimePrefix.through top := (PrimePrefix.mem_through top top).mpr ⟨by omega, prime⟩
        rw [complete] at member
        exact (PrimePrefix.mem_below top candidate).mp member |>.1
      let nextTop := if Nat.Prime candidate then candidate else top
      have nextPrime : Nat.Prime nextTop := by
        dsimp [nextTop]
        split <;> assumption
      have nextLarge : 3≤nextTop := by dsimp [nextTop]; split <;> omega
      have nextComplete : PrimePrefix.through nextTop=PrimePrefix.below (candidate+2) := by
        rw [PrimePrefix.below_two candidate odd (by omega)]
        dsimp [nextTop]
        by_cases candidatePrime : Nat.Prime candidate
        · simp [candidatePrime, PrimePrefix.through, PrimePrefix.below_succ]
        · simp [candidatePrime, complete]
      let nextEnv := setLocal env "candidate" (natural (candidate+2))
      obtain ⟨finalTop, finalPrime, finalLarge, finalReached, rest⟩ :=
        ih (candidate+2) nextTop (by omega) nextPrime nextLarge (by omega) nextComplete nextEnv
          (by simpa [nextEnv, setLocal] using getN) (by simpa [nextEnv, setLocal] using getPrimes) rfl
      have trialResult : Evaluates program env (.call (.name "is_prime_with_list") [.name "candidate", .name "primes"])
          (primeHeap base top) (flag (decide (Nat.Prime candidate))) (primeHeap base top) := by
        apply Evaluates.call (.cons (Evaluates.lookup getCandidate) (.cons (Evaluates.lookup getPrimes) .nil))
        have run := is_prime_with_list.result candidate (by omega) (PrimePrefix.through top)
          (by intro p member; have := (PrimePrefix.mem_through p top).mp member |>.2.two_le; omega)
          (primeHeap base top) base.length (by simp [Mutable.sequence, primeHeap, HeapCollection.found])
        simpa only [complete, PrimePrefix.trial_below candidate (by omega)] using run
      have augment : BlockReturns program env
          [.augAssign (.name "candidate") .add (.constant (.int 2)), .whileDo condition body []]
          (primeHeap base nextTop) (.scalar .none) (primeHeap base finalTop) := by
        apply BlockReturns.augAssign (value := natural (candidate+2))
          (middle := primeHeap base nextTop) (afterOperation := primeHeap base nextTop) getCandidate Evaluates.constant
        · simp [inplace, Mutable.binary, scalarBinary]
        · exact rest
      refine ⟨finalTop, finalPrime, finalLarge, finalReached, ?_⟩
      apply BlockReturns.while_loop (flag := true)
        (by simpa only [show top<n by omega, decide_true] using guard n top prime base env getN getPrimes) rfl
      by_cases candidatePrime : Nat.Prime candidate
      · apply BlockReturns.branch (flag := true) (by simpa only [candidatePrime, decide_true] using trialResult) rfl
        apply BlockReturns.discard (value := .scalar .none) (middle := primeHeap base nextTop)
        · have appendResult := Evaluates.append (module := program) (globals := []) (Evaluates.lookup getPrimes)
            (HeapCollection.found base (numbers (PrimePrefix.through top)) [])
            (Evaluates.lookup getCandidate) (HeapCollection.found base (numbers (PrimePrefix.through top)) [])
          have nextList : PrimePrefix.through nextTop=PrimePrefix.through top++[candidate] := by
            rw [show nextTop=candidate by simp [nextTop, candidatePrime]]
            change PrimePrefix.below (candidate+1)=PrimePrefix.through top++[candidate]
            rw [PrimePrefix.below_succ, if_pos candidatePrime, complete]
          simpa only [primeHeap, HeapCollection.store, nextList, numbers, List.map_append,
            List.map_cons, List.map_nil] using appendResult
        · simpa only [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            Bool.false_or, Bool.false_eq_true, if_false, List.nil_append] using augment
      · apply BlockReturns.branch (flag := false) (by simpa only [candidatePrime, decide_false] using trialResult) rfl
        simpa only [nextTop, if_neg candidatePrime, body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false, List.nil_append] using augment

/-- Extending a complete odd-prime prefix always terminates and preserves its shared address. -/
theorem result (n top : Nat) (prime : Nat.Prime top) (large : 3≤top) (base : Mutable.Heap) :
    ∃ finalTop, Nat.Prime finalTop ∧ 3≤finalTop ∧ n≤finalTop ∧
      ReturnsValue program "ensure_primes_up_to" [natural n, .ref base.length]
        (primeHeap base top) (.scalar .none) (primeHeap base finalTop) := by
  let initial := functionEnv ((findFunctionIn program.body "ensure_primes_up_to").getD ([], [])).1
    ((findFunctionIn program.body "ensure_primes_up_to").getD ([], [])).2 [natural n, .ref base.length] []
  have firstTest : Evaluates program initial
      (.compare (.subscript (.name "primes") (.unaryOp .uSub (.constant (.int 1)))) [(.gtE, .name "n")])
      (primeHeap base top) (flag (decide (n≤top))) (primeHeap base top) := by
    apply Evaluates.compare (av := natural top) (bv := natural n) (middle := primeHeap base top)
      (afterRight := primeHeap base top) (get_last top prime base initial rfl) (Evaluates.lookup rfl)
    change some (Except.ok (ε := ExceptionObject) (flag (decide ((n : Int)≤(top : Int)))), primeHeap base top)=_
    simp only [Int.ofNat_le]
  by_cases reached : n≤top
  · refine ⟨top, prime, large, reached, ?_⟩
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := true) (by simpa only [reached, decide_true] using firstTest) rfl
    exact BlockReturns.return_value Evaluates.constant
  · obtain ⟨q, beyond, qPrime⟩ := Nat.exists_infinite_primes (max n (top+2))
    have qLarge : 3≤q := by omega
    have qOdd := PrimePrefix.odd_prime q qPrime qLarge
    have topOdd := PrimePrefix.odd_prime top prime large
    have distance : top+2+2*((q-(top+2))/2)=q := by omega
    have complete : PrimePrefix.through top=PrimePrefix.below (top+2) := by
      rw [PrimePrefix.below_two top topOdd large]
      simp [PrimePrefix.through, PrimePrefix.below_succ, prime]
    obtain ⟨finalTop, finalPrime, finalLarge, finalReached, searched⟩ :=
      loop n q qPrime (by omega) ((q-(top+2))/2) (top+2) top distance prime large (by omega) complete base
        (setLocal initial "candidate" (natural (top+2)))
        (by simp [setLocal, initial]; rfl) (by simp [setLocal, initial]; rfl) rfl
    refine ⟨finalTop, finalPrime, finalLarge, finalReached, ?_⟩
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (flag := false) (by simpa only [reached, decide_false] using firstTest) rfl
    apply BlockReturns.assign (value := natural (top+2)) (middle := primeHeap base top)
    · exact Evaluates.binary (av := natural top) (bv := natural 2)
        (get_last top prime base initial rfl) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
    · exact searched
/-- A caller whose shared prefix already reaches the target takes the first return. -/
theorem already (n top : Nat) (prime : Nat.Prime top) (reached : n≤top) (base : Mutable.Heap) :
    ReturnsValue program "ensure_primes_up_to" [natural n, .ref base.length]
      (primeHeap base top) (.scalar .none) (primeHeap base top) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (value := flag true) (flag := true)
  · apply Evaluates.compare (av := natural top) (bv := natural n)
      (middle := primeHeap base top) (afterRight := primeHeap base top) (finalHeap := primeHeap base top)
      (get_last top prime base _ rfl) (Evaluates.lookup rfl)
    change some (Except.ok (ε := ExceptionObject) (flag (decide ((n : Int)≤(top : Int)))), primeHeap base top)=_
    simp only [Int.ofNat_le, reached, decide_true]
  · rfl
  · exact BlockReturns.return_value Evaluates.constant
end ensure_primes_up_to

namespace can_be_written_as_prime_plus_twice_square
private abbrev code : List Stmt :=
  executableBody ((findFunctionIn program.body "can_be_written_as_prime_plus_twice_square").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[1]? : Option Stmt) with | some (.forIn _ _ body _) => body | _ => []

private theorem loop (n : Nat) (items remaining : List Nat) (position address : Nat)
    (ordered : remaining.Pairwise (· ≤ ·)) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (sequence : Mutable.sequence (.ref address) heap=some (.ok (numbers items), heap))
    (suffix : items.drop position=remaining) (getN : env "n"=some (natural n)) :
    BlockReturns program env
      [.sequenceStepLoop "p" (.ref address) position body [], .return (.constant (.bool false))]
      heap (flag (remaining.any (PrimeTwiceSquare.square n))) heap := by
  induction remaining generalizing position env with
  | nil =>
      apply BlockReturns.sequence_done (source := .ref address) sequence
      · have missing : items[position]?=none := by
          simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
        simp [numbers, missing]
      · exact BlockReturns.return_value Evaluates.constant
  | cons p rest ih =>
      obtain ⟨afterFirst, restOrdered⟩ := List.pairwise_cons.mp ordered
      have item : items[position]?=some p := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      have suffixTail : items.drop (position+1)=rest := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := setLocal env "p" (natural p)
      have getP : bound "p"=some (natural p) := rfl
      have boundN : bound "n"=some (natural n) := by simpa [bound, setLocal] using getN
      have compared : Evaluates program bound (.compare (.name "p") [(.gtE, .name "n")])
          heap (flag (decide (n≤p))) heap := by
        apply Evaluates.compare (av := natural p) (bv := natural n) (middle := heap) (afterRight := heap)
          (Evaluates.lookup getP) (Evaluates.lookup boundN)
        change some (Except.ok (ε := ExceptionObject) (flag (decide ((n : Int)≤(p : Int)))), heap)=_
        simp only [Int.ofNat_le]
      apply BlockReturns.sequence_step (source := .ref address) (value := natural p) sequence
        (by simp [numbers, item])
      by_cases beyond : n≤p
      · have rejected : (p::rest).any (PrimeTwiceSquare.square n)=false := by
          apply List.any_eq_false.mpr
          intro q member
          have bound : n≤q := by
            rcases List.mem_cons.mp member with rfl | member
            · exact beyond
            · exact le_trans beyond (afterFirst q member)
          simp [PrimeTwiceSquare.square, show ¬q<n by omega]
        rw [rejected]
        apply BlockReturns.branch (flag := true) (by simpa only [beyond, decide_true] using compared) rfl
        apply BlockReturns.break_loop rfl
        exact BlockReturns.return_value Evaluates.constant
      · have before : p<n := by omega
        apply BlockReturns.branch (flag := false) (by simpa only [beyond, decide_false] using compared) rfl
        let rem := n-p
        let remainderEnv := setLocal bound "rem" (natural rem)
        have remainder : Evaluates program bound (.binOp (.name "n") .sub (.name "p")) heap (natural rem) heap :=
          Evaluates.binary (Evaluates.lookup boundN) (Evaluates.lookup getP)
            (by simp [Mutable.binary, scalarBinary, rem, Int.natCast_sub (show p≤n by omega)])
        have getRem : remainderEnv "rem"=some (natural rem) := rfl
        have oddTest : Evaluates program remainderEnv
            (.compare (.binOp (.name "rem") .mod (.constant (.int 2))) [(.notEq, .constant (.int 0))])
            heap (flag (decide (rem%2≠0))) heap := by
          apply Evaluates.compare (av := natural (rem%2)) (bv := natural 0) (middle := heap) (afterRight := heap)
          · exact Evaluates.binary (av := natural rem) (bv := natural 2)
              (Evaluates.lookup getRem) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
          · exact Evaluates.constant
          · change some (Except.ok (ε := ExceptionObject) (flag (decide (((rem%2 : Nat) : Int)≠0))), heap)=_
            simp only [ne_eq, Int.natCast_eq_zero]
        apply BlockReturns.assign remainder
        by_cases odd : rem%2≠0
        · have rejected : PrimeTwiceSquare.square n p=false := by
            change decide (p<n ∧ rem%2=0 ∧ _)=false
            simp [odd]
          rw [List.any_cons, rejected, Bool.false_or]
          apply BlockReturns.branch (flag := true) (by simpa [odd] using oddTest) rfl
          apply BlockReturns.continue_loop rfl
          exact ih (position+1) restOrdered remainderEnv suffixTail
            (by simpa [remainderEnv, bound, setLocal] using getN)
        · have even : rem%2=0 := by omega
          apply BlockReturns.branch (flag := false) (by simpa [even] using oddTest) rfl
          let t := rem/2
          let divided := setLocal remainderEnv "t" (natural t)
          let root := Nat.sqrt t
          let rooted := setLocal divided "s" (natural root)
          have quotient : Evaluates program remainderEnv (.binOp (.name "rem") .floorDiv (.constant (.int 2)))
              heap (natural t) heap :=
            Evaluates.binary (Evaluates.lookup getRem) Evaluates.constant
              (by simp [Mutable.binary, scalarBinary, t])
          have rootResult : Evaluates program divided (.call (.name "isqrt") [.name "t"])
              heap (natural root) heap := by
            apply Evaluates.library_call (.cons (Evaluates.lookup rfl) .nil) rfl
            have same : Math.isqrt t=Nat.sqrt t := Nat.eq_sqrt.mpr (Math.isqrt_spec t)
            change scalarResult (some (Math.isqrtCall [.int t])) heap=_
            simp [scalarResult, Math.isqrtCall, same, root, show ¬ (t : Int)<0 by omega]
          have tested : Evaluates program rooted
              (.compare (.binOp (.name "s") .mult (.name "s")) [(.eq, .name "t")])
              heap (flag (decide (root*root=t))) heap := by
            apply Evaluates.compare (av := natural (root*root)) (bv := natural t) (middle := heap) (afterRight := heap)
            · exact Evaluates.binary (av := natural root) (bv := natural root)
                (Evaluates.lookup rfl) (Evaluates.lookup rfl) (by simp [Mutable.binary, scalarBinary])
            · exact Evaluates.lookup (by simp [rooted, divided, setLocal])
            · change some (Except.ok (ε := ExceptionObject) (flag (decide (((root*root : Nat) : Int)=(t : Int)))), heap)=_
              simp only [Int.natCast_inj]
          have squareValue : PrimeTwiceSquare.square n p=decide (root*root=t) := by
            change decide (p<n ∧ rem%2=0 ∧ root*root=t)=decide (root*root=t)
            simp only [before, even, true_and]
          rw [List.any_cons, squareValue]
          apply BlockReturns.assign quotient
          apply BlockReturns.assign rootResult
          by_cases square : root*root=t
          · simp only [square, decide_true, Bool.true_or]
            apply BlockReturns.branch (flag := true) (by simpa only [square, decide_true] using tested) rfl
            exact BlockReturns.return_value Evaluates.constant
          · simp only [square, decide_false, Bool.false_or]
            apply BlockReturns.branch (flag := false) (by simpa only [square, decide_false] using tested) rfl
            apply BlockReturns.loop_resume
            exact ih (position+1) restOrdered rooted suffixTail
              (by simpa [rooted, divided, remainderEnv, bound, setLocal] using getN)

/-- Once the shared prefix reaches n, the source's remainder and isqrt tests are complete. -/
theorem result (n top : Nat) (prime : Nat.Prime top) (reached : n≤top) (base : Mutable.Heap) :
    ReturnsValue program "can_be_written_as_prime_plus_twice_square" [natural n, .ref base.length]
      (primeHeap base top) (flag (PrimeTwiceSquare.fast n)) (primeHeap base top) := by
  let initial := functionEnv ((findFunctionIn program.body "can_be_written_as_prime_plus_twice_square").getD ([], [])).1
    ((findFunctionIn program.body "can_be_written_as_prime_plus_twice_square").getD ([], [])).2
    [natural n, .ref base.length] []
  apply BlockReturns.function rfl rfl
  apply BlockReturns.discard (value := .scalar .none) (middle := primeHeap base top)
    (Evaluates.call (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil))
      (ensure_primes_up_to.already n top prime reached base))
  have sequence : Mutable.sequence (.ref base.length) (primeHeap base top)=
      some (.ok (numbers (PrimePrefix.through top)), primeHeap base top) := by
    simp [Mutable.sequence, primeHeap, HeapCollection.found]
  apply BlockReturns.for_sequence (by intro expressions h; cases h) (Evaluates.lookup rfl) sequence
  have run := loop n (PrimePrefix.through top) (PrimePrefix.through top) 0 base.length
    (PrimePrefix.ordered_below (top+1)) (primeHeap base top) initial sequence (by simp) rfl
  have same := PrimeTwiceSquare.fromPrimes_correct n top reached
  change (PrimePrefix.through top).any (PrimeTwiceSquare.square n)=PrimeTwiceSquare.fast n at same
  rw [same] at run
  exact run
end can_be_written_as_prime_plus_twice_square

namespace solve
open ProjectEulerSolutions.PrimeTwiceSquare ProjectEulerSolutions.PrimeTwiceSquareSearch
private abbrev code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private abbrev body : List Stmt :=
  match (code[2]? : Option Stmt) with | some (.whileDo _ body _) => body | _ => []

private theorem loop (fuel n top : Nat) (distance : n+2*fuel=winner) (lower : 9≤n) (odd : n%2=1)
    (prime : Nat.Prime top) (large : 3≤top) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n"=some (natural n)) (getPrimes : env "primes"=some (.ref base.length)) :
    ∃ finalHeap, BlockReturns program env [.whileDo (.constant (.bool true)) body []]
      (primeHeap base top) (natural winner) finalHeap := by
  induction fuel generalizing n top env with
  | zero =>
      have same : n=winner := by omega
      have accepted := winner_accepted
      have composite : ¬ Nat.Prime n := by
        have facts := Bool.and_eq_true_iff.mp accepted |>.1
        have facts := of_decide_eq_true facts
        exact same ▸ facts.2.1
      have absent : fast n=false := by
        have facts := Bool.and_eq_true_iff.mp accepted |>.2
        cases computed : fast winner with
        | false => simpa only [same] using computed
        | true => simp [computed] at facts
      obtain ⟨checkedTop, checkedPrime, checkedLarge, reached, ensured⟩ :=
        ensure_primes_up_to.result n top prime large base
      have primeCall : Evaluates program env (.unaryOp .not
          (.call (.name "is_prime_with_list") [.name "n", .name "primes"]))
          (primeHeap base checkedTop) (flag true) (primeHeap base checkedTop) := by
        apply Evaluates.unary
        · apply Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil))
          have run := is_prime_with_list.result n (by omega) (PrimePrefix.through checkedTop)
            (by intro p member; have := (PrimePrefix.mem_through p checkedTop).mp member |>.2.two_le; omega)
            (primeHeap base checkedTop) base.length (by simp [Mutable.sequence, primeHeap, HeapCollection.found])
          simpa only [PrimePrefix.trial_through n checkedTop (by omega) reached, composite, decide_false] using run
        · rfl
      have writtenCall : Evaluates program env (.unaryOp .not
          (.call (.name "can_be_written_as_prime_plus_twice_square") [.name "n", .name "primes"]))
          (primeHeap base checkedTop) (flag true) (primeHeap base checkedTop) := by
        apply Evaluates.unary
        · apply Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil))
          simpa only [absent] using can_be_written_as_prime_plus_twice_square.result n checkedTop checkedPrime reached base
        · rfl
      refine ⟨primeHeap base checkedTop, ?_⟩
      apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
      apply BlockReturns.discard (value := .scalar .none) (middle := primeHeap base checkedTop)
        (Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil)) ensured)
      apply BlockReturns.branch (flag := true) primeCall rfl
      apply BlockReturns.branch (flag := true) writtenCall rfl
      exact BlockReturns.return_value (Evaluates.lookup (by simpa only [same] using getN))
  | succ fuel ih =>
      obtain ⟨checkedTop, checkedPrime, checkedLarge, reached, ensured⟩ :=
        ensure_primes_up_to.result n top prime large base
      obtain ⟨finalHeap, rest⟩ := ih (n+2) checkedTop (by omega) (by omega) (by omega)
        checkedPrime checkedLarge (setLocal env "n" (natural (n+2))) rfl (by simpa [setLocal] using getPrimes)
      have primeCall : Evaluates program env (.unaryOp .not
          (.call (.name "is_prime_with_list") [.name "n", .name "primes"]))
          (primeHeap base checkedTop) (flag (!decide (Nat.Prime n))) (primeHeap base checkedTop) := by
        apply Evaluates.unary
        · apply Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil))
          have run := is_prime_with_list.result n (by omega) (PrimePrefix.through checkedTop)
            (by intro p member; have := (PrimePrefix.mem_through p checkedTop).mp member |>.2.two_le; omega)
            (primeHeap base checkedTop) base.length (by simp [Mutable.sequence, primeHeap, HeapCollection.found])
          simpa only [PrimePrefix.trial_through n checkedTop (by omega) reached] using run
        · rfl
      have augment : BlockReturns program env
          [.augAssign (.name "n") .add (.constant (.int 2)), .whileDo (.constant (.bool true)) body []]
          (primeHeap base checkedTop) (natural winner) finalHeap := by
        apply BlockReturns.augAssign (value := natural (n+2))
          (middle := primeHeap base checkedTop) (afterOperation := primeHeap base checkedTop) getN Evaluates.constant
        · simp [inplace, Mutable.binary, scalarBinary]
        · exact rest
      refine ⟨finalHeap, ?_⟩
      apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
      apply BlockReturns.discard (value := .scalar .none) (middle := primeHeap base checkedTop)
        (Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil)) ensured)
      by_cases primeN : Nat.Prime n
      · apply BlockReturns.branch (flag := false) (by simpa only [primeN, decide_true, Bool.not_true] using primeCall) rfl
        simpa only [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false, List.nil_append] using augment
      · have represented : fast n=true := by
          have rejected := no_earlier n (by omega)
          have composite : oddComposite n=true := by simp [oddComposite, odd, primeN, show 1<n by omega]
          cases written : fast n with
          | false => simp [counterexample, composite, written] at rejected
          | true => rfl
        have writtenCall : Evaluates program env (.unaryOp .not
            (.call (.name "can_be_written_as_prime_plus_twice_square") [.name "n", .name "primes"]))
            (primeHeap base checkedTop) (flag false) (primeHeap base checkedTop) := by
          apply Evaluates.unary
          · apply Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil))
            simpa only [represented] using can_be_written_as_prime_plus_twice_square.result n checkedTop checkedPrime reached base
          · rfl
        apply BlockReturns.branch (flag := true) (by simpa only [primeN, decide_false, Bool.not_false] using primeCall) rfl
        apply BlockReturns.branch (flag := false) writtenCall rfl
        simpa only [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false, List.nil_append] using augment

/-- The real mutable-list solver terminates at the mathematically certified least counterexample. -/
theorem result : ∃ finalHeap, ReturnsValue program "solve" [] [] (natural winner) finalHeap := by
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [] []
  let listed := setLocal initial "primes" (.ref 0)
  let ready := setLocal listed "n" (natural 9)
  obtain ⟨finalHeap, searched⟩ := loop 2884 9 3 rfl (by decide) (by decide) Nat.prime_three (by decide)
    [] ready (by simp [ready, setLocal]) (by simp [ready, listed, setLocal])
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  have allocated : Evaluates program initial (.list [.constant (.int 2), .constant (.int 3)])
      [] (.ref 0) (primeHeap [] 3) := by
    simp only [primeHeap, HeapCollection.working, PrimePrefix.initial, numbers, List.map_cons,
      List.map_nil, List.nil_append]
    exact Evaluates.list (.cons Evaluates.constant (.cons Evaluates.constant .nil))
  apply (BlockContinues.annAssign allocated).returns
  apply BlockReturns.assign Evaluates.constant
  exact searched
end solve

theorem solve.«Equiv(naive_default)» :
    (HeapCalls.scalarFunction program "solve" []).Equiv
      (HeapCalls.scalarFunction P46Ast.module "naive_default" []) := by
  obtain ⟨_, optimized⟩ := solve.result
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p46.naive_default.result []
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
      (HeapCalls.scalarFunction P46Ast.module "naive_default" []).Supported := by
  obtain ⟨_, optimized⟩ := solve.result
  exact ⟨(HeapCalls.ReturnsValue.scalar_returns (values := []) optimized).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p46.naive_default.«Supported»⟩

end solvers.«46»
