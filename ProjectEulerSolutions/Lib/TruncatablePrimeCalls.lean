import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerStatements.Lib.TruncatablePrimeBase

/-! Shared source-loop reasoning for the two prime-tree traversals. -/
namespace ProjectEulerSolutions.TruncatablePrimeCalls
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerStatements.TruncatablePrimeBase

set_option Elab.async false

abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
def numbers (items : List Nat) : List Mutable.Value := items.map number

def writes : List String := ["candidate", "total", "digit", "prefix"]
def Stable (before after : HeapCalls.Env) : Prop :=
  ∀ query, query ∉ writes → after query = before query

theorem stable_refl (env : HeapCalls.Env) : Stable env env := fun _ _ => rfl

theorem stable_trans {first middle last : HeapCalls.Env}
    (left : Stable first middle) (right : Stable middle last) : Stable first last := by
  intro query unchanged
  exact (right query unchanged).trans (left query unchanged)

def digitBody (predicateName : String) : List Stmt :=
  [.assign [.name "candidate"]
      (.binOp (.binOp (.name "base") .mult (.name "prefix")) .add (.name "digit")),
   .ifThen (.call (.name "is_prime") [.name "candidate"])
     [.expr (.call (.attribute (.name "following") "append") [.name "candidate"]),
      .ifThen (.call (.name predicateName) [.name "candidate", .name "base"])
        [.augAssign (.name "total") .add (.name "candidate")] []] []]

def selected (n : Nat) : List Nat :=
  if Nat.Prime n then [n] else []

def contribution (radix n : Nat) : Nat :=
  if Nat.Prime n ∧ leftCheck radix n = true then n else 0

variable (program : Module) (radix : Nat) (predicateName : String)

/-- Appending one prime child mutates only the fresh collector, and predicate allocations
extend the heap. The source still checks all nonzero candidate digits. -/
theorem digit_result (valid : 2 ≤ radix)
    (primeResult : ∀ n heap, ReturnsValue program "is_prime" [number n] heap
      (flag (decide (Nat.Prime n))) heap)
    (predicateResult : ∀ n, radix ≤ n → RightPrime radix n → ∀ heap,
      ∃ extra, ReturnsValue program predicateName [number n, number radix] heap
        (flag (leftCheck radix n)) (heap ++ extra))
    (parent digit total : Nat) (parentPositive : 0 < parent) (right : RightPrime radix parent)
    (digitSmall : digit < radix)
    (base : Mutable.Heap) (before : List Nat) (garbage : Mutable.Heap) (env : HeapCalls.Env)
    (getBase : env "base" = some (number radix)) (getParent : env "prefix" = some (number parent))
    (getDigit : env "digit" = some (number digit)) (getTotal : env "total" = some (number total))
    (getFollowing : env "following" = some (.ref base.length)) :
    ∃ finalEnv finalGarbage, BlockContinues program env (digitBody predicateName)
      (HeapCollection.working base (numbers before) garbage) finalEnv
      (HeapCollection.working base (numbers (before ++ selected (radix * parent + digit)))
        finalGarbage) ∧
      finalEnv "total" = some (number (total + contribution radix (radix * parent + digit))) ∧
      Stable env finalEnv ∧ finalEnv "prefix" = some (number parent) := by
  let candidate := radix * parent + digit
  let ready := setLocal env "candidate" (number candidate)
  let heap := HeapCollection.working base (numbers before) garbage
  have assigned : BlockContinues program env
      [(digitBody predicateName)[0]'(by simp [digitBody])] heap ready heap := by
    apply BlockContinues.assign
    exact Evaluates.binary (av := number (radix * parent)) (bv := number digit)
      (middle := heap) (afterRight := heap)
      (Evaluates.binary (av := number radix) (bv := number parent)
        (middle := heap) (afterRight := heap) (finalHeap := heap)
        (Evaluates.lookup getBase) (Evaluates.lookup getParent)
        (by simp [Mutable.binary, scalarBinary]))
      (Evaluates.lookup getDigit) (by simp [candidate, Mutable.binary, scalarBinary])
  have readyTotal : ready "total" = some (number total) := by simpa [ready, setLocal] using getTotal
  have readyFollowing : ready "following" = some (.ref base.length) := by
    simpa [ready, setLocal] using getFollowing
  have readyBase : ready "base" = some (number radix) := by simpa [ready, setLocal] using getBase
  have readyParent : ready "prefix" = some (number parent) := by simpa [ready, setLocal] using
    getParent
  have tested : Evaluates program ready (.call (.name "is_prime") [.name "candidate"]) heap
      (flag (decide (Nat.Prime candidate))) heap :=
    Evaluates.call (.cons (Evaluates.lookup (by simp [ready, setLocal])) .nil)
      (primeResult candidate heap)
  have readyStable : Stable env ready := by
    intro query unchanged
    simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at unchanged
    simp [ready, setLocal, unchanged.1]
  by_cases prime : Nat.Prime candidate
  · have parentEq : candidate / radix = parent := by
      dsimp only [candidate]
      rw [Nat.mul_add_div (by omega : 0 < radix), Nat.div_eq_of_lt digitSmall, Nat.add_zero]
    have childRight : RightPrime radix candidate := right_of_parent radix candidate prime
      (parentEq.symm ▸ right)
    have large : radix ≤ candidate := by dsimp only [candidate]; nlinarith
    let appended := HeapCollection.working base (numbers (before ++ [candidate])) garbage
    have appendCall : BlockContinues program ready
        [.expr (.call (.attribute (.name "following") "append") [.name "candidate"])] heap ready
          appended := by
      have receiver : Evaluates program ready (.name "following") heap (.ref base.length) heap :=
        Evaluates.lookup readyFollowing
      have item : Evaluates program ready (.name "candidate") heap (number candidate) heap :=
        Evaluates.lookup (by simp [ready, setLocal])
      have evaluated := Evaluates.append receiver (HeapCollection.found base (numbers before)
        garbage)
        item (HeapCollection.found base (numbers before) garbage)
      simpa only [heap, appended, HeapCollection.store, numbers, List.map_append,
        List.map_singleton] using BlockContinues.discard evaluated
    obtain ⟨extra, checked⟩ := predicateResult candidate large childRight appended
    have predicate : Evaluates program ready
        (.call (.name predicateName) [.name "candidate", .name "base"]) appended
        (flag (leftCheck radix candidate)) (appended ++ extra) :=
      Evaluates.call (.cons (Evaluates.lookup (by simp [ready, setLocal]))
        (.cons (Evaluates.lookup readyBase) .nil)) checked
    by_cases accepted : leftCheck radix candidate = true
    · let finalEnv := setLocal ready "total" (number (total + candidate))
      refine ⟨finalEnv, garbage ++ extra, ?_, ?_, ?_, ?_⟩
      · have done : BlockContinues program ready (digitBody predicateName).tail heap finalEnv
            (appended ++ extra) := by
          apply BlockContinues.branch (middle := heap) (flag := true) (value := flag true)
            (by simpa only [prime, decide_true] using tested) rfl
          apply appendCall.append
          apply BlockContinues.branch (flag := true) (value := flag true)
            (by simpa only [accepted] using predicate) rfl
          apply BlockContinues.augAssign (left := number total) (right := number candidate)
            readyTotal (Evaluates.lookup (by simp [ready, setLocal]))
          simp [inplace, Mutable.binary, scalarBinary]
        simpa [digitBody, selected, candidate, prime, appended, numbers,
          HeapCollection.append] using assigned.append done
      · simp [finalEnv, setLocal, contribution, candidate, prime, accepted]
      · apply stable_trans readyStable
        intro query unchanged
        simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at unchanged
        simp [finalEnv, setLocal, unchanged.2.1]
      · simpa [finalEnv, setLocal] using readyParent
    · refine ⟨ready, garbage ++ extra, ?_, ?_, readyStable, readyParent⟩
      · have done : BlockContinues program ready (digitBody predicateName).tail heap ready
            (appended ++ extra) := by
          apply BlockContinues.branch (middle := heap) (flag := true) (value := flag true)
            (by simpa only [prime, decide_true] using tested) rfl
          apply appendCall.append
          apply BlockContinues.branch (flag := false) (value := flag false)
            (by simpa only [Bool.eq_false_iff.mpr accepted] using predicate) rfl
          exact .nil
        simpa [digitBody, selected, candidate, prime, appended, numbers,
          HeapCollection.append] using assigned.append done
      · simpa [contribution, candidate, accepted] using readyTotal
  · refine ⟨ready, garbage, ?_, ?_, readyStable, readyParent⟩
    · have done : BlockContinues program ready (digitBody predicateName).tail heap ready heap := by
        apply BlockContinues.branch (middle := heap) (flag := false) (value := flag false)
          (by simpa only [prime, decide_false] using tested) rfl
        exact .nil
      simpa [digitBody, selected, candidate, prime] using assigned.append done
    · simpa [contribution, candidate, prime] using readyTotal

def weight (radix : Nat) (items : List Nat) : Nat :=
  (items.filter (leftCheck radix)).sum

theorem weight_append (radix : Nat) (first second : List Nat) :
    weight radix (first ++ second) = weight radix first + weight radix second := by
  simp [weight, List.filter_append, List.sum_append]

theorem contribution_eq_weight (radix n : Nat) :
    contribution radix n = weight radix (selected n) := by
  by_cases prime : Nat.Prime n <;> cases check : leftCheck radix n <;>
    simp [contribution, weight, selected, prime, check]

def expanded (radix parent first count : Nat) : List Nat :=
  (List.range' first count).flatMap fun digit => selected (radix * parent + digit)

theorem expanded_step (radix parent first count : Nat) :
    expanded radix parent first (count + 1) =
      selected (radix * parent + first) ++ expanded radix parent (first + 1) count := by
  simp [expanded, List.range'_succ]

theorem expanded_full (radix parent : Nat) :
    expanded radix parent 1 (radix - 1) = children radix parent := by
  have filtered (items : List Nat) :
      items.flatMap selected = items.filter (fun n => decide (Nat.Prime n)) := by
    induction items with
    | nil => rfl
    | cons n rest ih => by_cases prime : Nat.Prime n <;> simp [selected, prime, ih]
  simpa only [expanded, children, List.flatMap_map, Function.comp_def] using
    filtered ((List.range' 1 (radix - 1)).map (fun digit => radix * parent + digit))

theorem digit_loop (valid : 2 ≤ radix)
    (primeResult : ∀ n heap, ReturnsValue program "is_prime" [number n] heap
      (flag (decide (Nat.Prime n))) heap)
    (predicateResult : ∀ n, radix ≤ n → RightPrime radix n → ∀ heap,
      ∃ extra, ReturnsValue program predicateName [number n, number radix] heap
        (flag (leftCheck radix n)) (heap ++ extra))
    (parent first count total : Nat) (parentPositive : 0 < parent) (right : RightPrime radix parent)
    (firstPositive : 0 < first) (endpoint : first + count = radix)
    (base : Mutable.Heap) (before : List Nat) (garbage : Mutable.Heap) (env : HeapCalls.Env)
    (getBase : env "base" = some (number radix)) (getParent : env "prefix" = some (number parent))
    (getTotal : env "total" = some (number total))
    (getFollowing : env "following" = some (.ref base.length)) :
    ∃ finalEnv finalGarbage, BlockContinues program env
      [.rangeStepLoop "digit" first radix 1 (digitBody predicateName) []]
      (HeapCollection.working base (numbers before) garbage) finalEnv
      (HeapCollection.working base (numbers (before ++ expanded radix parent first count))
        finalGarbage) ∧
      finalEnv "total" = some (number (total + weight radix (expanded radix parent first count))) ∧
      Stable env finalEnv := by
  induction count generalizing first before garbage total env with
  | zero =>
      refine ⟨env, garbage, ?_, ?_, stable_refl env⟩
      · simpa only [expanded, List.range'_zero, List.flatMap_nil, List.append_nil] using
          (BlockContinues.range_done (module := program) (env := env)
            (heap := HeapCollection.working base (numbers before) garbage)
            (name := "digit") (next := first) (stop := radix) (stride := 1)
            (body := digitBody predicateName) (otherwise := [])
            (by simp [Calls.rangeContinues]; omega) .nil)
      · simpa [expanded, weight] using getTotal
  | succ count ih =>
      let bound := setLocal env "digit" (number first)
      obtain ⟨middleEnv, middleGarbage, executed, middleTotal, middleStable, middleParent⟩ :=
        digit_result program radix predicateName valid primeResult predicateResult parent first
          total
          parentPositive right (by omega) base before garbage bound
          (by simpa [bound, setLocal] using getBase) (by simpa [bound, setLocal] using getParent)
          (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getTotal)
          (by simpa [bound, setLocal] using getFollowing)
      have middleBase : middleEnv "base" = some (number radix) :=
        (middleStable "base" (by decide)).trans (by simpa [bound, setLocal] using getBase)
      have middleFollowing : middleEnv "following" = some (.ref base.length) :=
        (middleStable "following" (by decide)).trans (by simpa [bound, setLocal] using getFollowing)
      obtain ⟨finalEnv, finalGarbage, rest, finalTotal, finalStable⟩ := ih (first + 1)
        (total + contribution radix (radix * parent + first)) (by omega) (by omega)
        (before ++ selected (radix * parent + first)) middleGarbage middleEnv
        middleBase middleParent middleTotal middleFollowing
      refine ⟨finalEnv, finalGarbage, ?_, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [digitBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          expanded_step, List.append_assoc] using executed.append rest
      · simpa only [expanded_step, weight_append, ← contribution_eq_weight, Nat.add_assoc]
          using finalTotal
      · apply stable_trans (middle := bound)
        · intro query unchanged
          simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at unchanged
          simp [bound, setLocal, unchanged.2.2.1]
        · exact stable_trans middleStable finalStable

def parentBody (predicateName : String) : List Stmt :=
  [.forIn (.name "digit") (.call (.name "range") [.constant (.int 1), .name "base"])
    (digitBody predicateName) []]

theorem parent_loop (valid : 2 ≤ radix)
    (primeResult : ∀ n heap, ReturnsValue program "is_prime" [number n] heap
      (flag (decide (Nat.Prime n))) heap)
    (predicateResult : ∀ n, radix ≤ n → RightPrime radix n → ∀ heap,
      ∃ extra, ReturnsValue program predicateName [number n, number radix] heap
        (flag (leftCheck radix n)) (heap ++ extra))
    (items remaining : List Nat) (position address total : Nat)
    (parents : ∀ p ∈ remaining, 0 < p ∧ RightPrime radix p)
    (suffix : items.drop position = remaining)
    (base : Mutable.Heap) (before : List Nat) (garbage : Mutable.Heap) (env : HeapCalls.Env)
    (found : base[address]? = some (.list (numbers items)))
    (getBase : env "base" = some (number radix))
    (getTotal : env "total" = some (number total))
    (getFollowing : env "following" = some (.ref base.length)) :
    ∃ finalEnv finalGarbage, BlockContinues program env
      [.sequenceStepLoop "prefix" (.ref address) position (parentBody predicateName) []]
      (HeapCollection.working base (numbers before) garbage) finalEnv
      (HeapCollection.working base (numbers (before ++ remaining.flatMap (children radix)))
        finalGarbage) ∧
      finalEnv "total" = some (number (total + weight radix (remaining.flatMap (children radix)))) ∧
      Stable env finalEnv := by
  have sequence (values : List Nat) (extra : Mutable.Heap) :
      Mutable.sequence (.ref address) (HeapCollection.working base (numbers values) extra) =
        some (.ok (numbers items), HeapCollection.working base (numbers values) extra) := by
    have kept := HeapCollection.prefix_found base (.list (numbers values) :: extra) address _ found
    simp [Mutable.sequence, HeapCollection.working, kept]
  induction remaining generalizing position before garbage total env with
  | nil =>
      have missing : items[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      refine ⟨env, garbage, ?_, ?_, stable_refl env⟩
      · simpa only [List.flatMap_nil, List.append_nil] using
          (BlockContinues.sequence_done (module := program) (env := env)
            (source := .ref address) (items := numbers items) (name := "prefix") (position :=
              position)
            (body := parentBody predicateName) (otherwise := [])
            (sequence before garbage)
            (by simp [numbers, missing]) .nil)
      · simpa [weight] using getTotal
  | cons parent rest ih =>
      have item : items[position]? = some parent := by
        simpa [suffix] using (List.head?_drop (l := items) (i := position)).symm
      let bound := setLocal env "prefix" (number parent)
      obtain ⟨middleEnv, middleGarbage, scanned, middleTotal, middleStable⟩ :=
        digit_loop program radix predicateName valid primeResult predicateResult
          parent 1 (radix - 1) total (parents parent (by simp)).1 (parents parent (by simp)).2
          (by decide) (by omega) base before garbage bound
          (by simpa [bound, setLocal] using getBase) (by simp [bound, setLocal])
          (by simpa [bound, setLocal] using getTotal)
          (by simpa [bound, setLocal] using getFollowing)
      rw [expanded_full] at scanned middleTotal
      have executed : BlockContinues program bound (parentBody predicateName)
          (HeapCollection.working base (numbers before) garbage) middleEnv
          (HeapCollection.working base (numbers (before ++ children radix parent)) middleGarbage) :=
            by
        apply BlockContinues.for_range
          (.cons Evaluates.constant (.cons (Evaluates.lookup
            (by simpa [bound, setLocal] using getBase)) .nil)) rfl scanned
      have middleBase : middleEnv "base" = some (number radix) :=
        (middleStable "base" (by decide)).trans (by simpa [bound, setLocal] using getBase)
      have middleFollowing : middleEnv "following" = some (.ref base.length) :=
        (middleStable "following" (by decide)).trans (by simpa [bound, setLocal] using getFollowing)
      obtain ⟨finalEnv, finalGarbage, completed, finalTotal, finalStable⟩ :=
        ih (position + 1) (total + weight radix (children radix parent))
          (fun p member => parents p (by simp [member]))
          (by simp [List.drop_add_one_eq_tail_drop, suffix])
          (before ++ children radix parent) middleGarbage middleEnv
          middleBase middleTotal middleFollowing
      refine ⟨finalEnv, finalGarbage, ?_, ?_, ?_⟩
      · apply BlockContinues.sequence_step (source := .ref address) (items := numbers items)
          (value := number parent) (sequence before garbage) (by simp [numbers, item])
        simpa [parentBody, digitBody, Calls.loopContinuation, Calls.loopControl,
          Calls.loopControlStmt, List.append_assoc] using executed.append completed
      · simpa only [List.flatMap_cons, weight_append, Nat.add_assoc] using finalTotal
      · apply stable_trans (middle := bound)
        · intro query unchanged
          simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at unchanged
          simp [bound, setLocal, unchanged.2.2.2]
        · exact stable_trans middleStable finalStable

end ProjectEulerSolutions.TruncatablePrimeCalls
