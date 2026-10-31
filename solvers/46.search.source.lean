import solvers.«46.primes.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p46.py»
import ProjectEulerSolutions.Lib.PrimeTwiceSquareSuccessor
import lang.py.FormalSW.LoopProgress

/-! Source execution of the mutable prime-list search above any shared threshold. -/
set_option Elab.async false
namespace solvers.«46».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions ProjectEulerSolutions.PrimeTwiceSquare ProjectEulerSolutions.PrimeTwiceSquareSuccessor
private abbrev program := Solvers.P46Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev natural (n : Nat) : Mutable.Value := number n
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
private def numbers (items : List Nat) : List Mutable.Value := items.map natural
private abbrev primeHeap (base : Mutable.Heap) (top : Nat) : Mutable.Heap :=
  HeapCollection.working base (numbers (PrimePrefix.through top)) []
private def predicate : Expr := .boolOp .and
  [.compare (.name "n") [(.gt, .name "after")],
    .unaryOp .not (.call (.name "is_prime_with_list") [.name "n", .name "primes"]),
    .unaryOp .not (.call (.name "can_be_written_as_prime_plus_twice_square") [.name "n", .name "primes"])]
private def body : List Stmt :=
  [.expr (.call (.name "ensure_primes_up_to") [.name "n", .name "primes"]),
    .ifThen predicate [.return (.name "n")] [], .augAssign (.name "n") .add (.constant (.int 2))]
private def searchLoop : Stmt := .whileDo (.constant (.bool true)) body []

private theorem prepare (index top : Nat) (prime : Nat.Prime top) (large : 3 ≤ top)
    (base : Mutable.Heap) (env : HeapCalls.Env) (getN : env "n" = some (natural (value index)))
    (getPrimes : env "primes" = some (.ref base.length)) :
    ∃ checkedTop, Nat.Prime checkedTop ∧ 3 ≤ checkedTop ∧ value index ≤ checkedTop ∧
      BlockContinues program env (body.take 1) (primeHeap base top) env (primeHeap base checkedTop) := by
  obtain ⟨checkedTop, checkedPrime, checkedLarge, reached, ensured⟩ := ensure_primes_up_to.result (value index) top prime large base
  refine ⟨checkedTop, checkedPrime, checkedLarge, reached, ?_⟩
  exact BlockContinues.discard
    (Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil)) ensured)

private theorem tested (after : Int) (index top : Nat) (prime : Nat.Prime top) (reached : value index ≤ top)
    (base : Mutable.Heap) (env : HeapCalls.Env) (getAfter : env "after" = some (number after))
    (getN : env "n" = some (natural (value index))) (getPrimes : env "primes" = some (.ref base.length)) :
    Evaluates program env predicate (primeHeap base top) (flag (accepts after (value index))) (primeHeap base top) := by
  have above : Evaluates program env (.compare (.name "n") [(.gt, .name "after")])
      (primeHeap base top) (flag (decide (after < (value index : Int)))) (primeHeap base top) :=
    Evaluates.compare (Evaluates.lookup getN) (Evaluates.lookup getAfter) (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  have primal : Evaluates program env (.unaryOp .not (.call (.name "is_prime_with_list") [.name "n", .name "primes"]))
      (primeHeap base top) (flag (!decide (Nat.Prime (value index)))) (primeHeap base top) := by
    apply Evaluates.unary
    · apply Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil))
      have run := is_prime_with_list.result (value index) (by unfold value; omega) (PrimePrefix.through top)
        (by intro p member; have bound := (PrimePrefix.mem_through p top).mp member |>.2.two_le; omega)
        (primeHeap base top) base.length (by simp [Mutable.sequence, primeHeap, HeapCollection.found]; rfl)
      simpa only [PrimePrefix.trial_through (value index) top (by unfold value; omega) reached] using run
    · rfl
  have written : Evaluates program env (.unaryOp .not
      (.call (.name "can_be_written_as_prime_plus_twice_square") [.name "n", .name "primes"]))
      (primeHeap base top) (flag (!fast (value index))) (primeHeap base top) := by
    apply Evaluates.unary
    · exact Evaluates.call (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getPrimes) .nil))
        (can_be_written_as_prime_plus_twice_square.result (value index) top prime reached base)
    · rfl
  have odd : value index % 2 = 1 := by unfold value; omega
  have lower : 1 < value index := by unfold value; omega
  by_cases later : after < (value index : Int)
  · simp only [accepts, later, decide_true, Bool.true_and, counterexample, oddComposite, odd, lower, true_and, and_true]
    apply Evaluates.boolean_next (flag := true) (by simpa only [later, decide_true] using above) rfl rfl
    by_cases primeN : Nat.Prime (value index)
    · simp only [primeN, not_true_eq_false, decide_false, Bool.false_and]
      exact Evaluates.boolean_stop (flag := false) (by simpa only [primeN, decide_true, Bool.not_true] using primal) rfl rfl
    · simp only [primeN, not_false_eq_true, decide_true, Bool.true_and]
      apply Evaluates.boolean_next (flag := true) (by simpa only [primeN, decide_false, Bool.not_false] using primal) rfl rfl
      exact Evaluates.boolean_last written rfl
  · simp only [accepts, later, decide_false, Bool.false_and]
    exact Evaluates.boolean_stop (flag := false) (by simpa only [later, decide_false] using above) rfl rfl

/-- A rejected candidate extends the complete prime prefix and advances by two. -/
private theorem advance (after : Int) (index top : Nat) (rejected : accepts after (value index) = false)
    (prime : Nat.Prime top) (large : 3 ≤ top) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getAfter : env "after" = some (number after)) (getN : env "n" = some (natural (value index)))
    (getPrimes : env "primes" = some (.ref base.length)) :
    ∃ finalTop, Nat.Prime finalTop ∧ 3 ≤ finalTop ∧
      BlockContinues program env body (primeHeap base top)
        (setLocal env "n" (natural (value (index + 1)))) (primeHeap base finalTop) := by
  obtain ⟨checkedTop, checkedPrime, checkedLarge, reached, prepared⟩ := prepare index top prime large base env getN getPrimes
  have evaluated := tested after index checkedTop checkedPrime reached base env getAfter getN getPrimes
  rw [rejected] at evaluated
  have skipped : BlockContinues program env [body[1]'(by decide)] (primeHeap base checkedTop) env (primeHeap base checkedTop) :=
    BlockContinues.branch evaluated rfl BlockContinues.nil
  have incremented : BlockContinues program env [body[2]'(by decide)] (primeHeap base checkedTop)
      (setLocal env "n" (natural (value (index + 1)))) (primeHeap base checkedTop) := by
    apply BlockContinues.augAssign (left := natural (value index)) (right := natural 2) getN Evaluates.constant
    simp [inplace, Mutable.binary, scalarBinary, value, Nat.mul_add, Int.add_assoc]
  exact ⟨checkedTop, checkedPrime, checkedLarge, prepared.append (skipped.append incremented)⟩

private theorem accepted (after : Int) (index top : Nat) (selected : accepts after (value index) = true)
    (prime : Nat.Prime top) (large : 3 ≤ top) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getAfter : env "after" = some (number after)) (getN : env "n" = some (natural (value index)))
    (getPrimes : env "primes" = some (.ref base.length)) :
    ∃ finalTop, ∀ following, BlockReturns program env (body ++ following) (primeHeap base top)
      (natural (value index)) (primeHeap base finalTop) := by
  obtain ⟨checkedTop, checkedPrime, checkedLarge, reached, prepared⟩ := prepare index top prime large base env getN getPrimes
  have evaluated := tested after index checkedTop checkedPrime reached base env getAfter getN getPrimes
  rw [selected] at evaluated
  refine ⟨checkedTop, ?_⟩
  intro following
  apply prepared.returns
  apply BlockReturns.branch evaluated rfl
  exact BlockReturns.return_value (Evaluates.lookup getN)

private theorem loop (after : Int) (success : Success after) (index top : Nat) (bounded : index ≤ winnerIndex after success)
    (prime : Nat.Prime top) (large : 3 ≤ top) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getAfter : env "after" = some (number after)) (getN : env "n" = some (natural (value index)))
    (getPrimes : env "primes" = some (.ref base.length)) :
    ∃ finalHeap, BlockReturns program env [searchLoop] (primeHeap base top) (natural (answer after success)) finalHeap := by
  by_cases same : index = winnerIndex after success
  · have selected : accepts after (value index) = true := by simpa only [same, answer] using winner_accepted after success
    obtain ⟨finalTop, returned⟩ := accepted after index top selected prime large base env getAfter getN getPrimes
    refine ⟨primeHeap base finalTop, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    simpa only [same, answer, Calls.loopContinuation, show Calls.loopControl body = false from rfl,
      Bool.false_eq_true, if_false] using returned [searchLoop]
  · have before : index < winnerIndex after success := by omega
    obtain ⟨nextTop, nextPrime, nextLarge, advanced⟩ :=
      advance after index top (earlier_rejected after success index before) prime large base env getAfter getN getPrimes
    obtain ⟨finalHeap, returned⟩ := loop after success (index + 1) nextTop (by omega) nextPrime nextLarge base
      (setLocal env "n" (natural (value (index + 1))))
      (by simpa [setLocal] using getAfter) (by simp [setLocal]) (by simpa [setLocal] using getPrimes)
    refine ⟨finalHeap, ?_⟩
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    simpa only [Calls.loopContinuation, show Calls.loopControl body = false from rfl, Bool.false_eq_true, if_false] using
      advanced.returns returned
termination_by winnerIndex after success - index
decreasing_by omega

private def setup : List Stmt :=
  [.annAssign (.name "primes") (.subscript (.name "list") (.name "int"))
      (some (.list [.constant (.int 2), .constant (.int 3)])), .assign [.name "n"] (.constant (.int 9))]
private def entry (after : Int) : HeapCalls.Env :=
  functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [number after] []
private def initialEnv (after : Int) (base : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal (entry after) "primes" (.ref base.length)) "n" (natural 9)

private theorem initialize_result (after : Int) (base : Mutable.Heap) :
    BlockContinues program (entry after) setup base (initialEnv after base) (primeHeap base 3) := by
  have allocated : Evaluates program (entry after) (.list [.constant (.int 2), .constant (.int 3)])
      base (.ref base.length) (primeHeap base 3) := by
    simp only [primeHeap, HeapCollection.working, PrimePrefix.initial, numbers, List.map_cons, List.map_nil]
    exact Evaluates.list (.cons Evaluates.constant (.cons Evaluates.constant .nil))
  exact (BlockContinues.annAssign allocated).append (BlockContinues.assign Evaluates.constant)

/-- Every successful mathematical search executes the complete actual mutable-list function. -/
theorem result (after : Int) (success : Success after) (base : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number after] base (natural (answer after success)) finalHeap := by
  obtain ⟨finalHeap, returned⟩ := loop after success 0 3 (by omega) Nat.prime_three (by decide) base (initialEnv after base)
    (by simp [initialEnv, entry, setLocal]; rfl) (by simp [initialEnv, setLocal, value]) (by simp [initialEnv, setLocal])
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  exact (initialize_result after base).returns returned

/-- If no candidate succeeds, the prime cache and every source iteration keep making progress. -/
theorem diverges (after : Int) (absent : ¬ Success after) :
    (HeapCalls.scalarFunction program "solve" [.int after]).Diverges := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "solve").getD ([], [])).2
  let invariant : HeapCalls.Env → Mutable.Heap → Prop := fun env heap =>
    ∃ index top : Nat, Nat.Prime top ∧ 3 ≤ top ∧ env "after" = some (number after) ∧
      env "n" = some (natural (value index)) ∧ env "primes" = some (.ref 0) ∧ heap = primeHeap [] top
  apply scalar_while_diverges program "solve" [.int after] args sourceBody setup body [number after]
    rfl rfl rfl rfl invariant (initialEnv after []) (primeHeap [] 3)
  · exact initialize_result after []
  · exact ⟨0, 3, Nat.prime_three, by decide, by simp [initialEnv, entry, setLocal]; rfl,
      by simp [initialEnv, setLocal, value], by simp [initialEnv, setLocal], rfl⟩
  · intro env heap current
    obtain ⟨index, top, prime, large, getAfter, getN, getPrimes, rfl⟩ := current
    obtain ⟨finalTop, finalPrime, finalLarge, advanced⟩ :=
      advance after index top (no_success after absent index) prime large [] env getAfter getN getPrimes
    refine ⟨setLocal env "n" (natural (value (index + 1))), primeHeap [] finalTop, advanced,
      index + 1, finalTop, finalPrime, finalLarge, ?_, ?_, ?_, rfl⟩
    · simpa [setLocal] using getAfter
    · simp [setLocal]
    · simpa [setLocal] using getPrimes

end solvers.«46».solve
