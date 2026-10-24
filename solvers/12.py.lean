import «.formalsw».solvers.«12.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p12.py»
import FormalSW
import ProjectEulerStatements.Lib.PrimeTrialDivision
import ProjectEulerStatements.Lib.TriangularDivisors
import ProjectEulerStatements.Lib.FactorDivisors

namespace solvers.«12»
open FormalSW.PyAst
private abbrev program := Solvers.P12Ast.module
private def origin : String := (program.classOrigins.lookup "PrimeTable").getD ""

def primeValues (ps : List Nat) : List Mutable.Value :=
  ps.map fun p : Nat => .scalar (.int (p : Int))

/-- The table and its prime list retain distinct, stable heap addresses. -/
def TableView (heap : Mutable.Heap) (address : Nat) (ps : List Nat) : Prop :=
  ∃ storage, heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]) ∧
    heap[storage]? = some (.list (primeValues ps))

private def initialized (heap : Mutable.Heap) : Mutable.Heap :=
  ((heap ++ [Mutable.Object.instance origin "PrimeTable" []]) ++ [Mutable.Object.list (primeValues [2])]).set heap.length
    (.instance origin "PrimeTable" [("primes", .ref (heap.length + 1))])

/-- Construct the actual source class, including its annotated list field. -/
theorem primeTable.result (heap : Mutable.Heap) :
    HeapCalls.ReturnsValue program "PrimeTable" [] heap (.ref heap.length) (initialized heap) := by
  apply HeapCalls.ReturnsValue.constructor (origin := origin) rfl rfl rfl rfl rfl
  change HeapCalls.BlockReturns _ _ [_] _ _ _
  apply HeapCalls.BlockContinues.returns (body := [_]) (following := [])
  · apply HeapCalls.BlockContinues.annAttribute
    · exact HeapCalls.Evaluates.list (.cons HeapCalls.Evaluates.constant .nil)
    · exact HeapCalls.Evaluates.lookup rfl
    · apply Mutable.writeField_result (origin := origin) (className := "PrimeTable") (fields := [])
      · simp
      · decide
  · intro anchors stack
    apply HeapCalls.Reaches.next rfl
    simp only [initialized, primeValues, List.length_append, List.length_cons, List.length_nil]
    exact .refl

theorem initialized_eq (heap : Mutable.Heap) : initialized heap = heap ++
    [.instance origin "PrimeTable" [("primes", .ref (heap.length + 1))], .list (primeValues [2])] := by
  simp [initialized, List.append_assoc, List.set_append_right]

theorem primeTable.view (heap : Mutable.Heap) : TableView (initialized heap) heap.length [2] := by
  rw [initialized_eq]
  refine ⟨heap.length + 1, ?_, ?_⟩ <;> simp

private def primeMethod : List Arg × List Stmt :=
  (findFunctionIn ((program.findClassBody "PrimeTable").getD []) "_is_prime").getD ([], [])

private def primeIteration : List Stmt :=
  match (primeMethod.2[1]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

/-- The source scans the stored list, including its early break at the square root. -/
theorem isPrime.loop (n : Nat) (heap : Mutable.Heap) (storage : Nat) (ps : List Nat)
    (found : heap[storage]? = some (.list (primeValues ps)))
    (positive : ∀ p ∈ ps, 0 < p) (position : Nat) (env : HeapCalls.Env)
    (getX : env "x" = some (.scalar (.int n)))
    (getRoot : env "r" = some (.scalar (.int (Nat.sqrt n)))) :
    HeapCalls.BlockReturns program env
      [.sequenceStepLoop "p" (.ref storage) position primeIteration [], .return (.constant (.bool true))]
      heap (.scalar (.bool (ProjectEulerStatements.PrimeTrialDivision.trial n (ps.drop position)))) heap := by
  have sequence : Mutable.sequence (.ref storage) heap = some (.ok (primeValues ps), heap) := by
    simp [Mutable.sequence, found]
  by_cases active : position < ps.length
  · let p := ps[position]
    let boundEnv := HeapCalls.setLocal env "p" (.scalar (.int p))
    have item : (primeValues ps)[position]? = some (.scalar (.int p)) := by
      simp only [primeValues, List.getElem?_map, List.getElem?_eq_getElem active, Option.map_some, p]
    have pPositive : 0 < p := positive p (List.getElem_mem active)
    have tested : HeapCalls.Evaluates program boundEnv
        (.compare (.name "p") [(.gt, .name "r")]) heap
        (.scalar (.bool (decide (Nat.sqrt n < p)))) heap := by
      apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl)
        (HeapCalls.Evaluates.lookup (by simpa [boundEnv, HeapCalls.setLocal] using getRoot))
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    rw [List.drop_eq_getElem_cons active]
    change HeapCalls.BlockReturns _ _ _ _
      (.scalar (.bool (ProjectEulerStatements.PrimeTrialDivision.trial n (p :: ps.drop (position + 1))))) _
    rw [ProjectEulerStatements.PrimeTrialDivision.trial]
    by_cases stop : Nat.sqrt n < p
    · rw [if_pos stop]
      apply HeapCalls.BlockReturns.sequence_step (source := .ref storage) sequence item
      apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [stop] using tested) rfl
      exact HeapCalls.BlockReturns.break_loop rfl
        (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
    · rw [if_neg stop]
      have divisible : HeapCalls.Evaluates program boundEnv
          (.compare (.binOp (.name "x") .mod (.name "p")) [(.eq, .constant (.int 0))]) heap
          (.scalar (.bool (decide (n % p = 0)))) heap := by
        apply HeapCalls.Evaluates.compare (av := .scalar (.int (n % p))) (middle := heap)
          (HeapCalls.Evaluates.binary
            (HeapCalls.Evaluates.lookup (by simpa [boundEnv, HeapCalls.setLocal] using getX))
            (HeapCalls.Evaluates.lookup rfl) (by simp [Mutable.binary, scalarBinary, pPositive, pPositive.ne']))
          HeapCalls.Evaluates.constant
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, Int.natCast_dvd_natCast, Nat.dvd_iff_mod_eq_zero]
      apply HeapCalls.BlockReturns.sequence_step (source := .ref storage) sequence item
      apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [stop] using tested) rfl
      by_cases divides : n % p = 0
      · rw [if_pos divides]
        exact HeapCalls.BlockReturns.branch (flag := true) (by simpa [divides] using divisible) rfl
          (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
      · rw [if_neg divides]
        apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [divides] using divisible) rfl
        apply HeapCalls.BlockReturns.loop_resume
        exact isPrime.loop n heap storage ps found positive (position + 1) boundEnv
          (by simpa [boundEnv, HeapCalls.setLocal] using getX)
          (by simpa [boundEnv, HeapCalls.setLocal] using getRoot)
  · rw [List.drop_eq_nil_of_le (by omega : ps.length ≤ position)]
    exact HeapCalls.BlockReturns.sequence_done sequence
      (List.getElem?_eq_none (by simpa [primeValues] using (show ps.length ≤ position by omega)))
      (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
termination_by ps.length - position
decreasing_by omega

/-- Reading the prime-list field preserves every existing object. -/
theorem table.read (heap : Mutable.Heap) (address storage : Nat)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)])) :
    HeapCalls.getAttribute program (.ref address) "primes" heap = some (.ok (.ref storage), heap) := by
  have defined : program.findClass "PrimeTable" =
      some (origin, (program.findClassBody "PrimeTable").getD []) := rfl
  simp [HeapCalls.getAttribute, HeapCalls.instanceClass, object, defined]

/-- The bound method is the original source definition, with its receiver captured. -/
theorem isPrime.bound (heap : Mutable.Heap) (address : Nat) (ps : List Nat)
    (table : TableView heap address ps) :
    HeapCalls.bindMethod program (.ref address) "_is_prime" heap =
      some (.ok (.method (.ref address) primeMethod.1 primeMethod.2), heap) := by
  obtain ⟨storage, object, found⟩ := table
  have defined : program.findClass "PrimeTable" =
      some (origin, (program.findClassBody "PrimeTable").getD []) := rfl
  have method : findFunctionIn ((program.findClassBody "PrimeTable").getD []) "_is_prime" =
      some primeMethod := rfl
  simp [HeapCalls.bindMethod, object, HeapCalls.instanceClass, defined, method]
  rfl

/-- The actual method returns trial division's result without modifying the heap. -/
theorem isPrime.result (n : Nat) (heap : Mutable.Heap) (address : Nat) (ps : List Nat)
    (table : TableView heap address ps) (positive : ∀ p ∈ ps, 0 < p) :
    HeapCalls.MethodReturns program (.ref address) primeMethod.1 primeMethod.2
      [.scalar (.int n)] heap (.scalar (.bool (ProjectEulerStatements.PrimeTrialDivision.trial n ps))) heap := by
  obtain ⟨storage, object, found⟩ := table
  apply HeapCalls.MethodReturns.of_body rfl
  apply HeapCalls.BlockReturns.assign (value := .scalar (.int (Nat.sqrt n))) (middle := heap)
  · apply HeapCalls.Evaluates.library_call (.cons (HeapCalls.Evaluates.lookup rfl) .nil) rfl
    change HeapCalls.scalarResult (some (Math.isqrtCall [.int n])) heap = _
    have root : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
    simp [HeapCalls.scalarResult, Math.isqrtCall, root, show ¬ (n : Int) < 0 by omega]
  · apply HeapCalls.BlockReturns.for_sequence (items := primeValues ps) (by intro expressions; simp)
      (HeapCalls.Evaluates.field (HeapCalls.Evaluates.lookup rfl) (solvers.«12».table.read heap address storage object))
    · simp [Mutable.sequence, found]
    · simpa only [List.drop_zero] using isPrime.loop n heap storage ps found positive 0 _ rfl rfl

/-- A complete prime cache makes the method an exact primality decision. -/
theorem isPrime.correct (n : Nat) (heap : Mutable.Heap) (address : Nat) (ps : List Nat)
    (table : TableView heap address ps) (large : 2 ≤ n) (ordered : ps.Pairwise (· ≤ ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps n) :
    HeapCalls.MethodReturns program (.ref address) primeMethod.1 primeMethod.2
      [.scalar (.int n)] heap (.scalar (.bool (decide (Nat.Prime n)))) heap := by
  have positive (p : Nat) (member : p ∈ ps) : 0 < p := ((complete p).mp member).1.pos
  simpa only [ProjectEulerStatements.PrimeTrialDivision.trial_complete n ps large ordered complete] using
    isPrime.result n heap address ps table positive

private abbrev lastExpression : Expr :=
  .subscript (.attribute (.name "self") "primes") (.unaryOp .uSub (.constant (.int 1)))

/-- Negative indexing reads the last cached prime, without copying the list. -/
theorem table.last (heap : Mutable.Heap) (address : Nat) (ps : List Nat) (last : Nat)
    (table : TableView heap address ps) (lastEntry : ps.getLast? = some last)
    (env : HeapCalls.Env) (self : env "self" = some (.ref address)) :
    HeapCalls.Evaluates program env lastExpression heap (.scalar (.int last)) heap := by
  obtain ⟨storage, object, found⟩ := table
  have nonempty : ps ≠ [] := by intro empty; simp [empty] at lastEntry
  have positive : 0 < ps.length := List.length_pos_iff.mpr nonempty
  have shifted : (-1 + (ps.length : Int)).toNat = ps.length - 1 := by omega
  have indexed : Mutable.index (primeValues ps) (.scalar (.int (-1))) heap =
      some (.ok (ps.length - 1), heap) := by
    simp [Mutable.index, primeValues, nonempty, shifted]
  have item : (primeValues ps)[ps.length - 1]? = some (.scalar (.int last)) := by
    rw [List.getLast?_eq_getElem?] at lastEntry
    simp only [primeValues, List.getElem?_map, lastEntry, Option.map_some]
  apply HeapCalls.Evaluates.subscript (by intros; simp)
    (HeapCalls.Evaluates.field (HeapCalls.Evaluates.lookup self) (solvers.«12».table.read heap address storage object))
    (HeapCalls.Evaluates.unary HeapCalls.Evaluates.constant rfl)
  simp [Mutable.getItem, found, indexed, Except.map, item]

/-- Replacing the list's contents retains the table identity and every other heap cell. -/
theorem table.replace (heap : Mutable.Heap) (address storage : Nat) (ps replacement : List Nat)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps))) :
    TableView (heap.set storage (.list (primeValues replacement))) address replacement := by
  have distinct : address ≠ storage := by
    rintro rfl
    rw [object] at found
    cases found
  have valid : storage < heap.length := (List.getElem?_eq_some_iff.mp found).1
  refine ⟨storage, ?_, ?_⟩
  · simpa [List.getElem?_set, Ne.symm distinct] using object
  · simp [valid]

private def ensureMethod : List Arg × List Stmt :=
  (findFunctionIn ((program.findClassBody "PrimeTable").getD []) "ensure_up_to").getD ([], [])

/-- An already large enough cache returns immediately and preserves the heap. -/
theorem ensureUpTo.cached (n last : Nat) (heap : Mutable.Heap) (address : Nat) (ps : List Nat)
    (table : TableView heap address ps) (lastEntry : ps.getLast? = some last) (enough : n ≤ last) :
    HeapCalls.MethodReturns program (.ref address) ensureMethod.1 ensureMethod.2
      [.scalar (.int n)] heap (.scalar .none) heap := by
  apply HeapCalls.MethodReturns.of_body rfl
  apply HeapCalls.BlockReturns.branch (flag := true) (value := .scalar (.bool true)) (middle := heap)
  · apply HeapCalls.Evaluates.compare (solvers.«12».table.last heap address ps last table lastEntry _ rfl)
      (HeapCalls.Evaluates.lookup rfl)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, enough]
  · rfl
  · exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant

private def ensureIteration : List Stmt :=
  match ((executableBody ensureMethod.2)[3]? : Option Stmt) with
  | some (.whileDo _ body _) => body
  | _ => []

private theorem heap_set_same (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨valid, value⟩ := List.getElem?_eq_some_iff.mp found
  rw [← value, List.set_getElem_self valid]

/-- One source iteration tests the candidate, appends precisely the primes, then advances by two. -/
theorem ensureUpTo.step (candidate : Nat) (heap : Mutable.Heap) (address storage : Nat)
    (ps : List Nat) (env : HeapCalls.Env)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (self : env "self" = some (.ref address))
    (getCandidate : env "candidate" = some (.scalar (.int candidate)))
    (large : 2 ≤ candidate) (ordered : ps.Pairwise (· ≤ ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps candidate) :
    HeapCalls.BlockContinues program env ensureIteration heap
      (HeapCalls.setLocal env "candidate" (.scalar (.int (candidate + 2 : Nat))))
      (heap.set storage (.list (primeValues (if Nat.Prime candidate then ps ++ [candidate] else ps)))) := by
  have table : TableView heap address ps := ⟨storage, object, found⟩
  have tested : HeapCalls.Evaluates program env
      (.call (.attribute (.name "self") "_is_prime") [.name "candidate"]) heap
      (.scalar (.bool (decide (Nat.Prime candidate)))) heap :=
    HeapCalls.Evaluates.method self (by decide) (isPrime.bound heap address ps table)
      (.cons (HeapCalls.Evaluates.lookup getCandidate) .nil)
      (isPrime.correct candidate heap address ps table large ordered complete)
  have first : HeapCalls.BlockContinues program env [ensureIteration[0]'(by decide)] heap env
      (heap.set storage (.list (primeValues (if Nat.Prime candidate then ps ++ [candidate] else ps)))) := by
    by_cases prime : Nat.Prime candidate
    · rw [if_pos prime]
      apply HeapCalls.BlockContinues.branch (flag := true) (by simpa [prime] using tested) rfl
      apply HeapCalls.BlockContinues.discard
      simpa only [primeValues, List.map_append, List.map_cons, List.map_nil] using
        HeapCalls.Evaluates.append
          (HeapCalls.Evaluates.field (HeapCalls.Evaluates.lookup self) (solvers.«12».table.read heap address storage object))
          found (HeapCalls.Evaluates.lookup getCandidate) found
    · rw [if_neg prime, heap_set_same heap storage _ found]
      exact HeapCalls.BlockContinues.branch (flag := false) (by simpa [prime] using tested) rfl
        HeapCalls.BlockContinues.nil
  apply first.append
  apply HeapCalls.BlockContinues.augAssign getCandidate HeapCalls.Evaluates.constant
  simp [HeapCalls.inplace, Mutable.binary, scalarBinary]

private abbrev ensureLoop : Stmt :=
  .whileDo (.compare lastExpression [(.lt, .name "n")]) ensureIteration []

/-- The unbounded source loop terminates with a complete ordered cache reaching its target. -/
theorem ensureUpTo.loop (target bound candidate : Nat) (primeBound : Nat.Prime bound)
    (enough : target ≤ bound) (heap : Mutable.Heap) (address storage : Nat)
    (ps : List Nat) (last : Nat) (env : HeapCalls.Env)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (lastEntry : ps.getLast? = some last) (ordered : ps.Pairwise (· < ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps candidate)
    (large : 3 ≤ candidate) (odd : candidate % 2 = 1)
    (self : env "self" = some (.ref address))
    (getN : env "n" = some (.scalar (.int target)))
    (getCandidate : env "candidate" = some (.scalar (.int candidate))) :
    ∃ finalPrimes finalLast, finalPrimes.getLast? = some finalLast ∧ target ≤ finalLast ∧
      finalPrimes.Pairwise (· < ·) ∧
      ProjectEulerStatements.PrimeTrialDivision.CompleteBelow finalPrimes (finalLast + 1) ∧
      HeapCalls.BlockReturns program env [ensureLoop] heap (.scalar .none)
        (heap.set storage (.list (primeValues finalPrimes))) := by
  have weakOrder := ordered.imp (fun h => Nat.le_of_lt h)
  have table : TableView heap address ps := ⟨storage, object, found⟩
  have tested : HeapCalls.Evaluates program env (.compare lastExpression [(.lt, .name "n")]) heap
      (.scalar (.bool (decide (last < target)))) heap := by
    apply HeapCalls.Evaluates.compare (solvers.«12».table.last heap address ps last table lastEntry env self)
      (HeapCalls.Evaluates.lookup getN)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases short : last < target
  · let nextPrimes := if Nat.Prime candidate then ps ++ [candidate] else ps
    let nextLast := if Nat.Prime candidate then candidate else last
    let nextHeap := heap.set storage (.list (primeValues nextPrimes))
    let nextEnv := HeapCalls.setLocal env "candidate" (.scalar (.int (candidate + 2 : Nat)))
    have lastNext : nextPrimes.getLast? = some nextLast := by
      by_cases prime : Nat.Prime candidate <;> simp [nextPrimes, nextLast, prime, lastEntry]
    have distinct : storage ≠ address := by
      rintro rfl
      rw [object] at found
      cases found
    have nextObject : nextHeap[address]? =
        some (.instance origin "PrimeTable" [("primes", .ref storage)]) := by
      simpa [nextHeap, distinct] using object
    have nextFound : nextHeap[storage]? = some (.list (primeValues nextPrimes)) := by
      simp [nextHeap, (List.getElem?_eq_some_iff.mp found).1]
    have nextOrder := ProjectEulerStatements.PrimeTrialDivision.advance_ordered candidate ps ordered complete
    have nextComplete := ProjectEulerStatements.PrimeTrialDivision.advance_complete candidate ps large odd complete
    have bounded := ProjectEulerStatements.PrimeTrialDivision.candidate_le_prime ps last candidate target bound
      weakOrder lastEntry complete short enough primeBound
    obtain ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalOrder, finalComplete, rest⟩ :=
      ensureUpTo.loop target bound (candidate + 2) primeBound enough nextHeap address storage
        nextPrimes nextLast nextEnv nextObject nextFound lastNext nextOrder nextComplete
        (by omega) (by omega)
        (by simpa [nextEnv, HeapCalls.setLocal] using self)
        (by simpa [nextEnv, HeapCalls.setLocal] using getN) rfl
    refine ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalOrder, finalComplete, ?_⟩
    apply HeapCalls.BlockReturns.while_loop (flag := true) (by simpa [short] using tested) rfl
    have step := ensureUpTo.step candidate heap address storage ps env object found self getCandidate
      (by omega) weakOrder complete
    simpa only [nextHeap, List.set_set] using step.returns rest
  · refine ⟨ps, last, lastEntry, by omega, ordered,
      ProjectEulerStatements.PrimeTrialDivision.complete_at_last ps last candidate weakOrder lastEntry complete, ?_⟩
    rw [heap_set_same heap storage _ found]
    apply HeapCalls.BlockReturns.while_loop (flag := false) (by simpa [short] using tested) rfl
    intro anchors stack
    exact .next rfl .refl
termination_by bound + 2 - candidate
decreasing_by omega

/-- Initialization starts at the first odd integer after the last cached prime. -/
theorem ensureUpTo.initialize (n last : Nat) (heap : Mutable.Heap) (address : Nat) (ps : List Nat)
    (table : TableView heap address ps) (lastEntry : ps.getLast? = some last) (short : last < n)
    (env : HeapCalls.Env) (self : env "self" = some (.ref address))
    (getN : env "n" = some (.scalar (.int n))) :
    HeapCalls.BlockContinues program env ((executableBody ensureMethod.2).take 3) heap
      (HeapCalls.setLocal env "candidate"
        (.scalar (.int (ProjectEulerStatements.PrimeTrialDivision.nextCandidate last)))) heap := by
  let body := executableBody ensureMethod.2
  let assigned := HeapCalls.setLocal env "candidate" (.scalar (.int (last + 1 : Nat)))
  have first : HeapCalls.BlockContinues program env [body[0]'(by decide)] heap env heap := by
    apply HeapCalls.BlockContinues.branch (flag := false) (value := .scalar (.bool false)) (middle := heap)
    · apply HeapCalls.Evaluates.compare (solvers.«12».table.last heap address ps last table lastEntry env self)
        (HeapCalls.Evaluates.lookup getN)
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, show ¬ n ≤ last by omega]
    · rfl
    · exact HeapCalls.BlockContinues.nil
  have second : HeapCalls.BlockContinues program env [body[1]'(by decide)] heap assigned heap := by
    apply HeapCalls.BlockContinues.assign
    apply HeapCalls.Evaluates.binary (solvers.«12».table.last heap address ps last table lastEntry env self)
      HeapCalls.Evaluates.constant
    simp [Mutable.binary, scalarBinary]
  have evenTest : HeapCalls.Evaluates program assigned
      (.compare (.binOp (.name "candidate") .mod (.constant (.int 2))) [(.eq, .constant (.int 0))]) heap
      (.scalar (.bool (decide ((last + 1) % 2 = 0)))) heap := by
    apply HeapCalls.Evaluates.compare (av := .scalar (.int ((last + 1) % 2 : Nat))) (middle := heap)
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  have third : HeapCalls.BlockContinues program assigned [body[2]'(by decide)] heap
      (HeapCalls.setLocal env "candidate"
        (.scalar (.int (ProjectEulerStatements.PrimeTrialDivision.nextCandidate last)))) heap := by
    by_cases even : (last + 1) % 2 = 0
    · apply HeapCalls.BlockContinues.branch (flag := true) (by simpa [even] using evenTest) rfl
      have advanced : HeapCalls.BlockContinues program assigned
          [.augAssign (.name "candidate") .add (.constant (.int 1))] heap
          (HeapCalls.setLocal assigned "candidate" (.scalar (.int (last + 2 : Nat)))) heap := by
        apply HeapCalls.BlockContinues.augAssign (left := .scalar (.int (last + 1 : Nat))) rfl
          HeapCalls.Evaluates.constant
        simp [HeapCalls.inplace, Mutable.binary, scalarBinary, Int.add_assoc]
      have overwritten : HeapCalls.setLocal assigned "candidate" (.scalar (.int (last + 2 : Nat))) =
          HeapCalls.setLocal env "candidate" (.scalar (.int (last + 2 : Nat))) := by
        funext query
        by_cases same : query = "candidate" <;> simp [assigned, HeapCalls.setLocal, same]
      rw [overwritten] at advanced
      simpa only [ProjectEulerStatements.PrimeTrialDivision.nextCandidate, if_pos even] using advanced
    · apply HeapCalls.BlockContinues.branch (flag := false) (by simpa [even] using evenTest) rfl
      simpa only [ProjectEulerStatements.PrimeTrialDivision.nextCandidate, if_neg even] using
        (HeapCalls.BlockContinues.nil (module := program) (env := assigned) (heap := heap))
  exact first.append (second.append third)

/-- The complete source method terminates for every target and preserves cache completeness. -/
theorem ensureUpTo.result (n last : Nat) (heap : Mutable.Heap) (address storage : Nat) (ps : List Nat)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (lastEntry : ps.getLast? = some last) (ordered : ps.Pairwise (· < ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps (last + 1)) :
    ∃ finalPrimes finalLast, finalPrimes.getLast? = some finalLast ∧ n ≤ finalLast ∧
      finalPrimes.Pairwise (· < ·) ∧
      ProjectEulerStatements.PrimeTrialDivision.CompleteBelow finalPrimes (finalLast + 1) ∧
      HeapCalls.MethodReturns program (.ref address) ensureMethod.1 ensureMethod.2
        [.scalar (.int n)] heap (.scalar .none) (heap.set storage (.list (primeValues finalPrimes))) := by
  have table : TableView heap address ps := ⟨storage, object, found⟩
  by_cases enoughNow : n ≤ last
  · refine ⟨ps, last, lastEntry, enoughNow, ordered, complete, ?_⟩
    rw [heap_set_same heap storage _ found]
    exact ensureUpTo.cached n last heap address ps table lastEntry enoughNow
  · have lastPrime : Nat.Prime last := ((complete last).mp (List.mem_of_mem_getLast? lastEntry)).1
    obtain ⟨bound, enough, primeBound⟩ := Nat.exists_infinite_primes n
    let candidate := ProjectEulerStatements.PrimeTrialDivision.nextCandidate last
    have bounds := ProjectEulerStatements.PrimeTrialDivision.nextCandidate_spec last lastPrime.two_le
    have readyComplete := ProjectEulerStatements.PrimeTrialDivision.nextCandidate_complete last ps lastPrime complete
    let env := HeapCalls.argumentEnv ensureMethod.1 [.ref address, .scalar (.int n)]
    let ready := HeapCalls.setLocal env "candidate" (.scalar (.int candidate))
    obtain ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalOrder, finalComplete, loop⟩ :=
      ensureUpTo.loop n bound candidate primeBound enough heap address storage ps last ready object found
        lastEntry ordered readyComplete bounds.2.1 bounds.2.2 rfl rfl rfl
    refine ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalOrder, finalComplete, ?_⟩
    apply HeapCalls.MethodReturns.of_body rfl
    exact (ensureUpTo.initialize n last heap address ps table lastEntry (by omega) env rfl rfl).returns loop

private def factorMethod : List Arg × List Stmt :=
  (findFunctionIn ((program.findClassBody "PrimeTable").getD []) "divisor_count").getD ([], [])

private def factorIteration : List Stmt :=
  match ((executableBody factorMethod.2)[4]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private def stripLoop : Stmt :=
  match (factorIteration[1]? : Option Stmt) with
  | some (.ifThen _ body _) => body[1]?.getD (.expr (.constant .none))
  | _ => .expr (.constant .none)

/-- The inner source loop removes every copy of a prime and counts its exponent. -/
theorem divisorCount.strip (x p exponent : Nat) (positive : 0 < x) (prime : Nat.Prime p)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getX : env "x" = some (.scalar (.int x)))
    (getP : env "p" = some (.scalar (.int p)))
    (getExponent : env "exp" = some (.scalar (.int exponent))) :
    ∃ finalEnv, HeapCalls.BlockContinues program env [stripLoop] heap finalEnv heap ∧
      finalEnv "x" = some (.scalar (.int (ProjectEulerStatements.FactorDivisors.strip x p).1)) ∧
      finalEnv "exp" = some (.scalar (.int (exponent + (ProjectEulerStatements.FactorDivisors.strip x p).2 : Nat))) ∧
      ∀ name, name ≠ "x" → name ≠ "exp" → finalEnv name = env name := by
  have tested : HeapCalls.Evaluates program env
      (.compare (.binOp (.name "x") .mod (.name "p")) [(.eq, .constant (.int 0))]) heap
      (.scalar (.bool (decide (p ∣ x)))) heap := by
    apply HeapCalls.Evaluates.compare (av := .scalar (.int (x % p))) (middle := heap)
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getX) (HeapCalls.Evaluates.lookup getP)
        (by simp [Mutable.binary, scalarBinary, prime.pos, prime.ne_zero])) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, Int.natCast_dvd_natCast]
  by_cases divides : p ∣ x
  · let divided := HeapCalls.setLocal env "x" (.scalar (.int (x / p : Nat)))
    let advanced := HeapCalls.setLocal divided "exp" (.scalar (.int (exponent + 1 : Nat)))
    have first : HeapCalls.BlockContinues program env
        [.augAssign (.name "x") .floorDiv (.name "p")] heap divided heap := by
      apply HeapCalls.BlockContinues.augAssign getX (HeapCalls.Evaluates.lookup getP)
      simp [HeapCalls.inplace, Mutable.binary, scalarBinary, prime.pos, prime.ne_zero]
    have second : HeapCalls.BlockContinues program divided
        [.augAssign (.name "exp") .add (.constant (.int 1))] heap advanced heap := by
      apply HeapCalls.BlockContinues.augAssign
        (by simpa [divided, HeapCalls.setLocal] using getExponent) HeapCalls.Evaluates.constant
      simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
    have quotientPositive : 0 < x / p := Nat.div_pos (Nat.le_of_dvd positive divides) prime.pos
    obtain ⟨finalEnv, rest, finalX, finalExponent, preserved⟩ :=
      divisorCount.strip (x / p) p (exponent + 1) quotientPositive prime heap advanced
        (by simp [advanced, divided, HeapCalls.setLocal])
        (by simpa [advanced, divided, HeapCalls.setLocal] using getP) rfl
    have stripEq : ProjectEulerStatements.FactorDivisors.strip x p =
        ((ProjectEulerStatements.FactorDivisors.strip (x / p) p).1,
          (ProjectEulerStatements.FactorDivisors.strip (x / p) p).2 + 1) := by
      rw [ProjectEulerStatements.FactorDivisors.strip, dif_pos ⟨prime.one_lt, positive, divides⟩]
    refine ⟨finalEnv, ?_, ?_, ?_, ?_⟩
    · apply HeapCalls.BlockContinues.while_loop (flag := true) (by simpa [divides] using tested) rfl
      exact (first.append second).append rest
    · simpa only [stripEq] using finalX
    · simpa only [stripEq, Nat.add_assoc, Nat.add_comm 1] using finalExponent
    · intro name notX notExponent
      rw [preserved name notX notExponent]
      simp [advanced, divided, HeapCalls.setLocal, notX, notExponent]
  · have stripEq : ProjectEulerStatements.FactorDivisors.strip x p = (x, 0) := by
      rw [ProjectEulerStatements.FactorDivisors.strip, dif_neg (by tauto)]
    refine ⟨env, ?_, ?_, ?_, by intros; rfl⟩
    · exact HeapCalls.BlockContinues.while_loop (flag := false) (by simpa [divides] using tested) rfl
        HeapCalls.BlockContinues.nil
    · simpa only [stripEq] using getX
    · simpa only [stripEq, Nat.add_zero] using getExponent
termination_by x
decreasing_by exact Nat.div_lt_self positive prime.one_lt

private abbrev factorTail : List Stmt := (executableBody factorMethod.2).drop 5

/-- After trial division the source accounts for the final residue and returns its accumulator. -/
theorem divisorCount.finish (x total : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getX : env "x" = some (.scalar (.int x)))
    (getTotal : env "total" = some (.scalar (.int total))) :
    HeapCalls.BlockReturns program env factorTail heap
      (.scalar (.int (total * ProjectEulerStatements.FactorDivisors.finish x : Nat))) heap := by
  have tested : HeapCalls.Evaluates program env (.compare (.name "x") [(.gt, .constant (.int 1))]) heap
      (.scalar (.bool (decide (1 < x)))) heap := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup getX) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases large : 1 < x
  · simp only [ProjectEulerStatements.FactorDivisors.finish, if_pos large]
    apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [large] using tested) rfl
    apply HeapCalls.BlockReturns.augAssign (value := .scalar (.int (total * 2 : Nat))) (afterOperation := heap)
      getTotal HeapCalls.Evaluates.constant
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
    · exact HeapCalls.BlockReturns.return_value (HeapCalls.Evaluates.lookup rfl)
  · simp only [ProjectEulerStatements.FactorDivisors.finish, if_neg large, Nat.mul_one]
    exact HeapCalls.BlockReturns.branch (flag := false) (by simpa [large] using tested) rfl
      (HeapCalls.BlockReturns.return_value (HeapCalls.Evaluates.lookup getTotal))

/-- The source's prime-list loop implements trial division and preserves the entire heap. -/
theorem divisorCount.loop (x total : Nat) (positive : 0 < x) (heap : Mutable.Heap)
    (storage : Nat) (ps : List Nat) (found : heap[storage]? = some (.list (primeValues ps)))
    (primes : ∀ p ∈ ps, Nat.Prime p) (position : Nat) (env : HeapCalls.Env)
    (getX : env "x" = some (.scalar (.int x)))
    (getTotal : env "total" = some (.scalar (.int total))) :
    HeapCalls.BlockReturns program env
      (.sequenceStepLoop "p" (.ref storage) position factorIteration [] :: factorTail) heap
      (.scalar (.int (total * ProjectEulerStatements.FactorDivisors.count x (ps.drop position) : Nat))) heap := by
  have sequence : Mutable.sequence (.ref storage) heap = some (.ok (primeValues ps), heap) := by
    simp [Mutable.sequence, found]
  by_cases active : position < ps.length
  · let p := ps[position]
    let boundEnv := HeapCalls.setLocal env "p" (.scalar (.int p))
    have prime : Nat.Prime p := primes p (List.getElem_mem active)
    have item : (primeValues ps)[position]? = some (.scalar (.int p)) := by
      simp only [primeValues, List.getElem?_map, List.getElem?_eq_getElem active, Option.map_some, p]
    have stopTest : HeapCalls.Evaluates program boundEnv
        (.compare (.binOp (.name "p") .mult (.name "p")) [(.gt, .name "x")]) heap
        (.scalar (.bool (decide (x < p * p)))) heap := by
      apply HeapCalls.Evaluates.compare (av := .scalar (.int (p * p : Nat))) (middle := heap)
        (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup rfl) (HeapCalls.Evaluates.lookup rfl)
          (by simp [Mutable.binary, scalarBinary]))
        (HeapCalls.Evaluates.lookup (by simpa [boundEnv, HeapCalls.setLocal] using getX))
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      norm_cast
    rw [List.drop_eq_getElem_cons active]
    change HeapCalls.BlockReturns _ _ _ _
      (.scalar (.int (total * ProjectEulerStatements.FactorDivisors.count x (p :: ps.drop (position + 1)) : Nat))) _
    rw [ProjectEulerStatements.FactorDivisors.count]
    by_cases stop : x < p * p
    · rw [if_pos stop]
      apply HeapCalls.BlockReturns.sequence_step (source := .ref storage) sequence item
      apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [stop] using stopTest) rfl
      apply HeapCalls.BlockReturns.break_loop rfl
      exact divisorCount.finish x total heap boundEnv
        (by simpa [boundEnv, HeapCalls.setLocal] using getX)
        (by simpa [boundEnv, HeapCalls.setLocal] using getTotal)
    · rw [if_neg stop]
      have dividesTest : HeapCalls.Evaluates program boundEnv
          (.compare (.binOp (.name "x") .mod (.name "p")) [(.eq, .constant (.int 0))]) heap
          (.scalar (.bool (decide (p ∣ x)))) heap := by
        apply HeapCalls.Evaluates.compare (av := .scalar (.int (x % p))) (middle := heap)
          (HeapCalls.Evaluates.binary
            (HeapCalls.Evaluates.lookup (by simpa [boundEnv, HeapCalls.setLocal] using getX))
            (HeapCalls.Evaluates.lookup rfl)
            (by simp [Mutable.binary, scalarBinary, prime.pos, prime.ne_zero])) HeapCalls.Evaluates.constant
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, Int.natCast_dvd_natCast]
      apply HeapCalls.BlockReturns.sequence_step (source := .ref storage) sequence item
      apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [stop] using stopTest) rfl
      by_cases divides : p ∣ x
      · apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [divides] using dividesTest) rfl
        apply HeapCalls.BlockReturns.assign (value := .scalar (.int 0)) (middle := heap) HeapCalls.Evaluates.constant
        let initialized := HeapCalls.setLocal boundEnv "exp" (.scalar (.int 0))
        obtain ⟨stripped, stripping, resultX, resultExponent, saved⟩ := divisorCount.strip x p 0 positive prime heap initialized
          (by simpa [initialized, boundEnv, HeapCalls.setLocal] using getX)
          (by simp [initialized, boundEnv, HeapCalls.setLocal]) rfl
        have totalSaved : stripped "total" = some (.scalar (.int total)) :=
          (saved "total" (by decide) (by decide)).trans
            (by simpa [initialized, boundEnv, HeapCalls.setLocal] using getTotal)
        apply HeapCalls.BlockContinues.returns (body := [stripLoop]) stripping
        apply HeapCalls.BlockReturns.augAssign
          (right := .scalar (.int ((ProjectEulerStatements.FactorDivisors.strip x p).2 + 1 : Nat)))
          (middle := heap)
          (value := .scalar (.int (total * ((ProjectEulerStatements.FactorDivisors.strip x p).2 + 1) : Nat)))
          (afterOperation := heap)
          totalSaved
        · apply HeapCalls.Evaluates.binary
            (HeapCalls.Evaluates.lookup (by simpa only [Nat.zero_add] using resultExponent)) HeapCalls.Evaluates.constant
          simp [Mutable.binary, scalarBinary]
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
        · apply HeapCalls.BlockReturns.loop_resume
          simpa only [Nat.mul_assoc] using divisorCount.loop
            (ProjectEulerStatements.FactorDivisors.strip x p).1
            (total * ((ProjectEulerStatements.FactorDivisors.strip x p).2 + 1))
            (ProjectEulerStatements.FactorDivisors.strip_spec x p positive prime).1
            heap storage ps found primes (position + 1) _
            (by simpa [HeapCalls.setLocal] using resultX) rfl
      · have unchanged : ProjectEulerStatements.FactorDivisors.strip x p = (x, 0) := by
          rw [ProjectEulerStatements.FactorDivisors.strip, dif_neg (by tauto)]
        simp only [unchanged, Nat.zero_add, Nat.one_mul]
        apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [divides] using dividesTest) rfl
        apply HeapCalls.BlockReturns.loop_resume
        exact divisorCount.loop x total positive heap storage ps found primes (position + 1) boundEnv
          (by simpa [boundEnv, HeapCalls.setLocal] using getX)
          (by simpa [boundEnv, HeapCalls.setLocal] using getTotal)
  · rw [List.drop_eq_nil_of_le (by omega : ps.length ≤ position)]
    apply HeapCalls.BlockReturns.sequence_done (source := .ref storage) (items := primeValues ps) sequence
      (List.getElem?_eq_none (by simpa [primeValues] using (show ps.length ≤ position by omega)))
    exact divisorCount.finish x total heap env getX getTotal
termination_by ps.length - position
decreasing_by all_goals omega

theorem ensureUpTo.bound (heap : Mutable.Heap) (address : Nat) (ps : List Nat)
    (table : TableView heap address ps) :
    HeapCalls.bindMethod program (.ref address) "ensure_up_to" heap =
      some (.ok (.method (.ref address) ensureMethod.1 ensureMethod.2), heap) := by
  obtain ⟨storage, object, found⟩ := table
  have defined : program.findClass "PrimeTable" =
      some (origin, (program.findClassBody "PrimeTable").getD []) := rfl
  have method : findFunctionIn ((program.findClassBody "PrimeTable").getD []) "ensure_up_to" =
      some ensureMethod := rfl
  simp [HeapCalls.bindMethod, object, HeapCalls.instanceClass, defined, method]
  rfl

/-- The source computes the exact divisor count while retaining a complete ordered cache. -/
theorem divisorCount.result (m last : Nat) (positive : 0 < m)
    (heap : Mutable.Heap) (address storage : Nat) (ps : List Nat)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (lastEntry : ps.getLast? = some last) (ordered : ps.Pairwise (· < ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps (last + 1)) :
    ∃ finalPrimes finalLast, finalPrimes.getLast? = some finalLast ∧
      finalPrimes.Pairwise (· < ·) ∧
      ProjectEulerStatements.PrimeTrialDivision.CompleteBelow finalPrimes (finalLast + 1) ∧
      HeapCalls.MethodReturns program (.ref address) factorMethod.1 factorMethod.2
        [.scalar (.int m)] heap (.scalar (.int m.divisors.card))
        (heap.set storage (.list (primeValues finalPrimes))) := by
  let env := HeapCalls.argumentEnv factorMethod.1 [.ref address, .scalar (.int m)]
  have guard : HeapCalls.Evaluates program env (.compare (.name "m") [(.ltE, .constant (.int 1))]) heap
      (.scalar (.bool (decide (m ≤ 1)))) heap := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases small : m ≤ 1
  · have equal : m = 1 := by omega
    refine ⟨ps, last, lastEntry, ordered, complete, ?_⟩
    rw [heap_set_same heap storage _ found]
    apply HeapCalls.MethodReturns.of_body rfl
    apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [small] using guard) rfl
    simp only [equal, Nat.divisors_one, Finset.card_singleton]
    exact HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant
  · obtain ⟨finalPrimes, finalLast, finalEntry, enough, finalOrder, finalComplete, ensured⟩ :=
      ensureUpTo.result (Nat.sqrt m + 1) last heap address storage ps object found lastEntry ordered complete
    let nextHeap := heap.set storage (.list (primeValues finalPrimes))
    have distinct : storage ≠ address := by rintro rfl; rw [object] at found; cases found
    have nextObject : nextHeap[address]? =
        some (.instance origin "PrimeTable" [("primes", .ref storage)]) := by
      simpa [nextHeap, distinct] using object
    have nextFound : nextHeap[storage]? = some (.list (primeValues finalPrimes)) := by
      simp [nextHeap, (List.getElem?_eq_some_iff.mp found).1]
    have primes (p : Nat) (member : p ∈ finalPrimes) : Nat.Prime p := ((finalComplete p).mp member).1
    have correct : ProjectEulerStatements.FactorDivisors.count m finalPrimes = m.divisors.card := by
      apply ProjectEulerStatements.FactorDivisors.count_correct m finalPrimes positive
        (finalOrder.imp (fun h => Nat.le_of_lt h)) primes
      intro p prime bounded _
      exact (finalComplete p).mpr ⟨prime, by omega⟩
    have argument : HeapCalls.Evaluates program env
        (.binOp (.call (.name "isqrt") [.name "m"]) .add (.constant (.int 1))) heap
        (.scalar (.int (Nat.sqrt m + 1 : Nat))) heap := by
      apply HeapCalls.Evaluates.binary (av := .scalar (.int (Nat.sqrt m))) (middle := heap)
        (bv := .scalar (.int 1)) (afterRight := heap)
      · apply HeapCalls.Evaluates.library_call (.cons (HeapCalls.Evaluates.lookup rfl) .nil) rfl
        change HeapCalls.scalarResult (some (Math.isqrtCall [.int m])) heap = _
        have root : Math.isqrt m = Nat.sqrt m := Nat.eq_sqrt.mpr (Math.isqrt_spec m)
        simp [HeapCalls.scalarResult, Math.isqrtCall, root, show ¬ (m : Int) < 0 by omega]
      · exact HeapCalls.Evaluates.constant
      · simp [Mutable.binary, scalarBinary]
    refine ⟨finalPrimes, finalLast, finalEntry, finalOrder, finalComplete, ?_⟩
    apply HeapCalls.MethodReturns.of_body rfl
    apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [small] using guard) rfl
    apply HeapCalls.BlockReturns.discard
      (HeapCalls.Evaluates.method rfl (by decide)
        (ensureUpTo.bound heap address ps ⟨storage, object, found⟩) (.cons argument .nil) ensured)
    apply HeapCalls.BlockReturns.assign (value := .scalar (.int m)) (middle := nextHeap)
      (HeapCalls.Evaluates.lookup rfl)
    apply HeapCalls.BlockReturns.assign (value := .scalar (.int 1)) (middle := nextHeap) HeapCalls.Evaluates.constant
    apply HeapCalls.BlockReturns.for_sequence (items := primeValues finalPrimes) (by intros; simp)
      (HeapCalls.Evaluates.field (HeapCalls.Evaluates.lookup rfl)
        (table.read nextHeap address storage nextObject))
    · simp [Mutable.sequence, nextFound]
    · simpa only [List.drop_zero, Nat.one_mul, correct] using
        divisorCount.loop m 1 positive nextHeap storage finalPrimes nextFound primes 0 _ rfl rfl

/-- A live table stores precisely the ordered primes through its last entry. -/
def CacheView (heap : Mutable.Heap) (address storage : Nat) : Prop :=
  ∃ ps last,
    heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]) ∧
    heap[storage]? = some (.list (primeValues ps)) ∧ ps.getLast? = some last ∧
    ps.Pairwise (· < ·) ∧ ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps (last + 1)

theorem primeTable.cache (heap : Mutable.Heap) :
    CacheView (initialized heap) heap.length (heap.length + 1) := by
  rw [initialized_eq]
  refine ⟨[2], 2, by simp, by simp, rfl, by simp, ?_⟩
  exact ProjectEulerStatements.PrimeTrialDivision.initial

theorem divisorCount.bound (heap : Mutable.Heap) (address storage : Nat)
    (cache : CacheView heap address storage) :
    HeapCalls.bindMethod program (.ref address) "divisor_count" heap =
      some (.ok (.method (.ref address) factorMethod.1 factorMethod.2), heap) := by
  obtain ⟨ps, last, object, found, lastEntry, ordered, complete⟩ := cache
  have defined : program.findClass "PrimeTable" =
      some (origin, (program.findClassBody "PrimeTable").getD []) := rfl
  have method : findFunctionIn ((program.findClassBody "PrimeTable").getD []) "divisor_count" =
      some factorMethod := rfl
  simp [HeapCalls.bindMethod, object, HeapCalls.instanceClass, defined, method]
  rfl

/-- A source call returns the exact count and leaves a valid cache for the next call. -/
theorem divisorCount.call (m : Nat) (positive : 0 < m) (heap : Mutable.Heap) (address storage : Nat)
    (cache : CacheView heap address storage) (env : HeapCalls.Env) (receiver : String) (expression : Expr)
    (lookup : env receiver = some (.ref address))
    (argument : HeapCalls.Evaluates program env expression heap (.scalar (.int m)) heap) :
    ∃ finalHeap, CacheView finalHeap address storage ∧
      HeapCalls.Evaluates program env (.call (.attribute (.name receiver) "divisor_count") [expression])
        heap (.scalar (.int m.divisors.card)) finalHeap := by
  have bound := divisorCount.bound heap address storage cache
  obtain ⟨ps, last, object, found, lastEntry, ordered, complete⟩ := cache
  obtain ⟨nextPrimes, nextLast, nextEntry, nextOrder, nextComplete, execution⟩ :=
    divisorCount.result m last positive heap address storage ps object found lastEntry ordered complete
  let finalHeap := heap.set storage (.list (primeValues nextPrimes))
  have distinct : storage ≠ address := by rintro rfl; rw [object] at found; cases found
  refine ⟨finalHeap, ⟨nextPrimes, nextLast, ?_, ?_, nextEntry, nextOrder, nextComplete⟩, ?_⟩
  · simpa [finalHeap, distinct] using object
  · simp [finalHeap, (List.getElem?_eq_some_iff.mp found).1]
  · exact HeapCalls.Evaluates.method lookup (by decide) bound (.cons argument .nil) execution

private def searchBody : List Stmt :=
  match ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[2]? : Option Stmt) with
  | some (.whileDo _ body _) => body
  | _ => []

private def splitEnv (env : HeapCalls.Env) (n : Nat) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal env "a" (.scalar (.int (ProjectEulerStatements.P12.splitLeft n))))
    "b" (.scalar (.int (ProjectEulerStatements.P12.splitRight n)))

/-- The source splits the triangular number into its two coprime factors. -/
theorem search.split (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (.scalar (.int n))) :
    HeapCalls.BlockContinues program env [searchBody[0]'(by decide)] heap (splitEnv env n) heap := by
  have tested : HeapCalls.Evaluates program env
      (.compare (.binOp (.name "n") .mod (.constant (.int 2))) [(.eq, .constant (.int 0))]) heap
      (.scalar (.bool (decide (n % 2 = 0)))) heap := by
    apply HeapCalls.Evaluates.compare (av := .scalar (.int (n % 2))) (middle := heap)
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getN) HeapCalls.Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  by_cases even : n % 2 = 0
  · apply HeapCalls.BlockContinues.branch (flag := true) (by simpa [even] using tested) rfl
    apply HeapCalls.BlockContinues.append
      (body := [.assign [.name "a"] (.binOp (.name "n") .floorDiv (.constant (.int 2)))])
      (middleEnv := HeapCalls.setLocal env "a" (.scalar (.int (n / 2 : Nat)))) (middleHeap := heap)
    · apply HeapCalls.BlockContinues.assign
      exact HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getN) HeapCalls.Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])
    · simp only [splitEnv, ProjectEulerStatements.P12.splitLeft, ProjectEulerStatements.P12.splitRight, if_pos even]
      apply HeapCalls.BlockContinues.assign
      exact HeapCalls.Evaluates.binary
        (HeapCalls.Evaluates.lookup (by simpa [HeapCalls.setLocal] using getN)) HeapCalls.Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])
  · apply HeapCalls.BlockContinues.branch (flag := false) (by simpa [even] using tested) rfl
    apply HeapCalls.BlockContinues.append (body := [.assign [.name "a"] (.name "n")])
      (middleEnv := HeapCalls.setLocal env "a" (.scalar (.int n))) (middleHeap := heap)
    · exact HeapCalls.BlockContinues.assign (HeapCalls.Evaluates.lookup getN)
    · simp only [splitEnv, ProjectEulerStatements.P12.splitLeft, ProjectEulerStatements.P12.splitRight, if_neg even]
      apply HeapCalls.BlockContinues.assign
      apply HeapCalls.Evaluates.binary (av := .scalar (.int (n + 1 : Nat))) (middle := heap)
        (bv := .scalar (.int 2)) (afterRight := heap)
      · exact HeapCalls.Evaluates.binary
          (HeapCalls.Evaluates.lookup (by simpa [HeapCalls.setLocal] using getN)) HeapCalls.Evaluates.constant
          (by simp [Mutable.binary, scalarBinary])
      · exact HeapCalls.Evaluates.constant
      · simp [Mutable.binary, scalarBinary]

private abbrev searchLoop : List Stmt := [.whileDo (.constant (.bool true)) searchBody []]

private theorem search.triangle (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (.scalar (.int n))) :
    HeapCalls.Evaluates program env
      (.binOp (.binOp (.name "n") .mult (.binOp (.name "n") .add (.constant (.int 1))))
        .floorDiv (.constant (.int 2))) heap (.scalar (.int (ProjectEulerStatements.P12.triangle n))) heap := by
  apply HeapCalls.Evaluates.binary (av := .scalar (.int (n * (n + 1) : Nat))) (middle := heap)
    (bv := .scalar (.int 2)) (afterRight := heap)
  · apply HeapCalls.Evaluates.binary (middle := heap) (afterRight := heap) (HeapCalls.Evaluates.lookup getN)
      (HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getN) HeapCalls.Evaluates.constant
        (value := .scalar (.int (n + 1 : Nat))) (afterRight := heap) (by simp [Mutable.binary, scalarBinary]))
    simp [Mutable.binary, scalarBinary]
  · exact HeapCalls.Evaluates.constant
  · simp [Mutable.binary, scalarBinary, ProjectEulerStatements.P12.triangle]

/-- The source search advances to the least qualifying triangular number, with no cutoff. -/
theorem search.loop (k n : Nat) (positive : 0 < n)
    (before : n ≤ Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k))
    (heap : Mutable.Heap) (address storage : Nat) (cache : CacheView heap address storage)
    (env : HeapCalls.Env) (getN : env "n" = some (.scalar (.int n)))
    (getK : env "k" = some (.scalar (.int k))) (getTable : env "pt" = some (.ref address)) :
    ∃ finalHeap, HeapCalls.BlockReturns program env searchLoop heap
      (.scalar (.int (ProjectEulerStatements.P12.firstTriangleOver k))) finalHeap := by
  let a := ProjectEulerStatements.P12.splitLeft n
  let b := ProjectEulerStatements.P12.splitRight n
  let t := ProjectEulerStatements.P12.triangle n
  let d := ProjectEulerStatements.P12.divisorCount t
  let parted := splitEnv env n
  let withD := HeapCalls.setLocal parted "d" (.scalar (.int d))
  have splitPositive := ProjectEulerStatements.P12.split_positive n positive
  obtain ⟨heap1, cache1, left⟩ := divisorCount.call a splitPositive.1 heap address storage cache parted "pt" (.name "a")
    (by simpa [parted, splitEnv, HeapCalls.setLocal] using getTable) (HeapCalls.Evaluates.lookup rfl)
  obtain ⟨heap2, cache2, right⟩ := divisorCount.call b splitPositive.2 heap1 address storage cache1 parted "pt" (.name "b")
    (by simpa [parted, splitEnv, HeapCalls.setLocal] using getTable) (HeapCalls.Evaluates.lookup rfl)
  have product : a.divisors.card * b.divisors.card = d :=
    (ProjectEulerStatements.P12.divisorCount_triangle_split n).symm
  have calculated : HeapCalls.Evaluates program parted
      (.binOp (.call (.attribute (.name "pt") "divisor_count") [.name "a"]) .mult
        (.call (.attribute (.name "pt") "divisor_count") [.name "b"])) heap (.scalar (.int d)) heap2 := by
    apply HeapCalls.Evaluates.binary left right
    simp [Mutable.binary, scalarBinary]
    exact_mod_cast product
  have tested : HeapCalls.Evaluates program withD (.compare (.name "d") [(.gt, .name "k")]) heap2
      (.scalar (.bool (decide (k < d)))) heap2 := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl)
      (HeapCalls.Evaluates.lookup (by simpa [withD, parted, splitEnv, HeapCalls.setLocal] using getK))
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases qualifies : k < d
  · have least := Nat.find_min' (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) qualifies
    have same : n = Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) := by omega
    refine ⟨heap2, ?_⟩
    apply HeapCalls.BlockReturns.while_loop (flag := true) HeapCalls.Evaluates.constant rfl
    apply HeapCalls.BlockContinues.returns (body := [searchBody[0]'(by decide)]) (search.split n heap env getN)
    apply HeapCalls.BlockReturns.assign calculated
    apply HeapCalls.BlockReturns.branch (flag := true) (by simpa [qualifies] using tested) rfl
    apply HeapCalls.BlockReturns.return_value
    simpa only [ProjectEulerStatements.P12.firstTriangleOver, ← same] using search.triangle n heap2 withD
      (by simpa [withD, parted, splitEnv, HeapCalls.setLocal] using getN)
  · have earlier : n < Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) := by
      have witness := Nat.find_spec (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k)
      by_contra late
      have same : n = Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) := by omega
      apply qualifies
      simpa only [d, t, same] using witness
    let advanced := HeapCalls.setLocal withD "n" (.scalar (.int (n + 1 : Nat)))
    obtain ⟨finalHeap, rest⟩ := search.loop k (n + 1) (by omega) (by omega) heap2 address storage cache2 advanced rfl
      (by simpa [advanced, withD, parted, splitEnv, HeapCalls.setLocal] using getK)
      (by simpa [advanced, withD, parted, splitEnv, HeapCalls.setLocal] using getTable)
    refine ⟨finalHeap, ?_⟩
    apply HeapCalls.BlockReturns.while_loop (flag := true) HeapCalls.Evaluates.constant rfl
    apply HeapCalls.BlockContinues.returns (body := [searchBody[0]'(by decide)]) (search.split n heap env getN)
    apply HeapCalls.BlockReturns.assign calculated
    apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [qualifies] using tested) rfl
    apply HeapCalls.BlockReturns.augAssign
      (value := .scalar (.int (n + 1 : Nat))) (afterOperation := heap2)
      (by simpa [withD, parted, splitEnv, HeapCalls.setLocal] using getN) HeapCalls.Evaluates.constant
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
    · exact rest
termination_by Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) - n
decreasing_by omega

/-- The optimized Python function returns the same mathematical least triangle as the reference. -/
theorem search.result (k : Nat) (heap : Mutable.Heap) :
    ∃ finalHeap, HeapCalls.ReturnsValue program "solve" [.scalar (.int k)] heap
      (.scalar (.int (ProjectEulerStatements.P12.naive k))) finalHeap := by
  let env := HeapCalls.setLocal (HeapCalls.setLocal
    (HeapCalls.argumentEnv [⟨"k", some (.name "int"), none⟩] [.scalar (.int k)])
    "pt" (.ref heap.length)) "n" (.scalar (.int 1))
  obtain ⟨finalHeap, loop⟩ := search.loop k 1 (by omega)
    (ProjectEulerStatements.P12.firstIndex_positive k) (initialized heap) heap.length (heap.length + 1)
    (primeTable.cache heap) env rfl rfl rfl
  refine ⟨finalHeap, ?_⟩
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.assign (HeapCalls.Evaluates.call .nil (primeTable.result heap))
  apply HeapCalls.BlockReturns.assign HeapCalls.Evaluates.constant
  simpa only [ProjectEulerStatements.P12.naive_eq_firstTriangleOver] using loop

theorem solve.«Equiv(naive)» (k : Int)
    (domain : Calls.Condition program (fun name => if name = "k" then some (.int k) else none)
      Solvers.P12Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int k]).Equiv
      (HeapCalls.scalarFunction P12Ast.module "naive" [.int k]) := by
  have nonnegative : 0 ≤ k := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare domain rfl
  lift k to Nat using nonnegative
  obtain ⟨heap, executed⟩ := search.result k []
  obtain ⟨extra, referenceRun⟩ := data.«project-euler-statements».data.py.project_euler_statements.p12.naive.result k []
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := [.int k]) executed
  have reference := HeapCalls.ReturnsValue.scalar_finishes (values := [.int k]) referenceRun
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem solve.«Supported» :
    (∀ k : Int, Calls.Condition program (fun name => if name = "k" then some (.int k) else none)
      Solvers.P12Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int k]).Supported) ∧
    (∀ k : Int, Calls.Condition program (fun name => if name = "k" then some (.int k) else none)
      Solvers.P12Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P12Ast.module "naive" [.int k]).Supported) := by
  constructor
  · intro k domain
    have nonnegative : 0 ≤ k := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare domain rfl
    lift k to Nat using nonnegative
    obtain ⟨heap, executed⟩ := search.result k []
    exact (HeapCalls.ReturnsValue.scalar_returns (values := [.int k]) executed).supported
  · intro k _
    exact data.«project-euler-statements».data.py.project_euler_statements.p12.naive.«Supported» k

end solvers.«12»
