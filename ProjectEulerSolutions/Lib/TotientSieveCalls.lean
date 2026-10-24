import ProjectEulerSolutions.Lib.TotientSieveOuterCalls

/-! Initialized source termination and totient-table correctness for every signed bound. -/
namespace ProjectEulerSolutions.TotientSieveCalls
open FormalSW.PyAst HeapCalls ProjectEulerStatements.TotientLinearSieve
set_option Elab.async false
variable {program : Module}

abbrev integer (n : Int) : Mutable.Value := .scalar (.int n)
abbrev sieveArgs : List Arg :=
  [⟨"limit", some (.name "int"), none⟩]
abbrev sizeExpression : Expr := .binOp (.name "limit") .add (.constant (.int 1))
abbrev phiExpression : Expr := .binOp (.list [.constant (.int 0)]) .mult sizeExpression
abbrev marksExpression : Expr := .binOp (.list [.constant (.bool false)]) .mult sizeExpression
abbrev smallCondition : Expr := .compare (.name "limit") [(.lt, .constant (.int 1))]

def phiHeap (base : Mutable.Heap) (size : Nat) : Mutable.Heap :=
  base ++ [.list [number 0], .list (entries size (fun _ => number 0))]

def marksHeap (base : Mutable.Heap) (size : Nat) : Mutable.Heap :=
  phiHeap base size ++ [.list [.scalar (.bool false)],
    .list (entries size (fun _ => .scalar (.bool false)))]

abbrev sieveCode : List Stmt :=
  [.assign [.name "phi"] phiExpression,
    .ifThen smallCondition [.return (.name "phi")] [],
    .assign [.name "is_comp"] marksExpression,
    .annAssign (.name "primes") (.subscript (.name "list") (.name "int")) (some (.list [])),
    .assign [.subscript (.name "phi") (.constant (.int 1))] (.constant (.int 1)),
    .forIn (.name "i") (.call (.name "range") [.constant (.int 2), sizeExpression]) outerBody [],
    .return (.name "phi")]

/-- A named function has the specified signed interface and executable sieve body. -/
def Implements (program : Module) (name : String) : Prop :=
  ∃ body, findFunctionIn program.body name = some (sieveArgs, body) ∧
    executableBody body = sieveCode

theorem allocate_phi (limit : Int) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (integer limit)) :
    Evaluates program env phiExpression base (.ref (base.length + 1))
      (phiHeap base (limit + 1).toNat) := by
  let scratch := base ++ [Mutable.Object.list [number 0]]
  apply Evaluates.binary (av := .ref base.length) (bv := integer (limit + 1))
    (middle := scratch) (afterRight := scratch)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary])
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, scratch, phiHeap, entries, Except.map]

theorem allocate_marks (limit : Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (number limit)) :
    Evaluates program env marksExpression (phiHeap base (limit + 1))
      (.ref (base.length + 3)) (marksHeap base (limit + 1)) := by
  let scratch := phiHeap base (limit + 1) ++ [Mutable.Object.list [.scalar (.bool false)]]
  apply Evaluates.binary (av := .ref ((phiHeap base (limit + 1)).length))
    (bv := number (limit + 1)) (middle := scratch) (afterRight := scratch)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary])
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
      Mutable.array, Mutable.allocate, number, scratch, marksHeap, phiHeap, entries, Except.map,
      List.append_assoc]

theorem allocate_cache (size : Nat) (base : Mutable.Heap) (env : HeapCalls.Env) :
    Evaluates program env (.list []) (marksHeap base size) (.ref (base.length + 4))
      (working base size (fun _ => 0) (fun _ => false) []) := by
  simpa [marksHeap, phiHeap, working, markValue, List.append_assoc] using
    (Evaluates.list (module := program) (env := env) (heap := marksHeap base size) .nil)

theorem small_result (limit : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (integer limit)) :
    Evaluates program env smallCondition heap (.scalar (.bool (decide (limit < 1)))) heap :=
  Evaluates.compare (middle := heap) (afterRight := heap)
    (Evaluates.lookup getLimit) Evaluates.constant
    (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])

theorem initial_heap (base : Mutable.Heap) (size : Nat) :
    working base size (Function.update (fun _ => 0) 1 1) (fun _ => false) [] =
      realize base size (expected 2) [] := by
  have first : Function.update (fun _ => 0) 1 1 = fun n => (expected 2 n).1 := by
    funext n
    by_cases same : n = 1 <;> simp [expected_initial, Function.update_apply, same]
  have second : (fun _ : Nat => false) = fun n => (expected 2 n).2 := by
    funext n
    simp only [expected_initial]
    split <;> rfl
  rw [first, second]

/-- Every positive bound runs the initialized source loops to completion. -/
theorem positive_result (name : String) (implementation : Implements program name)
    (limit : Nat) (positive : 1 ≤ limit) (base : Mutable.Heap) :
    ∃ finalCache, ReturnsValue program name [number limit] base
      (.ref (base.length + 1))
      (realize base (limit + 1) (expected (limit + 1)) finalCache) := by
  obtain ⟨body, found, executed⟩ := implementation
  let entry := HeapCalls.argumentEnv sieveArgs [number limit]
  let phiEnv := setLocal entry "phi" (.ref (base.length + 1))
  let marksEnv := setLocal phiEnv "is_comp" (.ref (base.length + 3))
  let cacheEnv := setLocal marksEnv "primes" (.ref (base.length + 4))
  have stable : Stable limit base.length cacheEnv := by constructor <;> rfl
  have complete : ∀ p, p ∈ ([] : List Nat) ↔ Nat.Prime p ∧ p < 2 := by
    intro p
    simp only [List.not_mem_nil, false_iff, not_and]
    intro prime
    have large := prime.two_le
    omega
  obtain ⟨finalEnv, finalCache, scanned, finalStable⟩ := outer_loop_result (program := program)
    limit 2 []
    (by decide) (by omega) (by simp) complete base cacheEnv stable
  have returned := scanned.returns
    (BlockReturns.return_value (rest := []) (Evaluates.lookup finalStable.getPhi))
  have initialized : BlockContinues program cacheEnv
      [.assign [.subscript (.name "phi") (.constant (.int 1))] (.constant (.int 1))]
      (working base (limit + 1) (fun _ => 0) (fun _ => false) []) cacheEnv
      (realize base (limit + 1) (expected 2) []) := by
    apply BlockContinues.setItem (by intro a b c impossible; cases impossible) Evaluates.constant
      (.cons (Evaluates.lookup stable.getPhi) (.cons Evaluates.constant .nil))
    have written := set_list (working base (limit + 1) (fun _ => 0) (fun _ => false) [])
      (base.length + 1) 1 (limit + 1) (fun _ => number 0) (number 1) (by omega)
      (phi_found base (limit + 1) (fun _ => 0) (fun _ => false) [])
    have same : Function.update (fun _ => number 0) 1 (number 1) =
        fun n => number (Function.update (fun _ => 0) 1 1 n) := by
      funext n
      by_cases same : n = 1 <;> simp [Function.update_apply, same]
    simpa only [same, store_phi, initial_heap] using written
  refine ⟨finalCache, ?_⟩
  apply BlockReturns.function found rfl
  rw [executed]
  apply BlockReturns.assign (value := .ref (base.length + 1))
    (by simpa only [integer, number, Int.natCast_add, Int.natCast_one, Int.toNat_natCast] using
      allocate_phi (program := program) limit base entry rfl)
  apply BlockReturns.branch (flag := false)
    (by simpa only [show ¬ (limit : Int) < 1 by omega, decide_false] using
      small_result (program := program) limit (phiHeap base (limit + 1)) phiEnv rfl) rfl
  apply BlockReturns.assign (allocate_marks (program := program) limit base phiEnv rfl)
  apply (BlockContinues.annAssign (allocate_cache (program := program)
    (limit + 1) base marksEnv)).returns
  apply initialized.returns
  apply BlockReturns.for_range (values := [number 2, number (limit + 1)])
    (start := 2) (stop := (limit + 1 : Nat)) (stride := 1)
    (middle := realize base (limit + 1) (expected 2) [])
    (.cons Evaluates.constant (.cons
      (Evaluates.binary (Evaluates.lookup stable.getLimit) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) .nil)) rfl
  simpa only [number, Nat.cast_add, Nat.cast_one] using returned

/-- All signed inputs terminate; their returned list contains exactly the requested totients. -/
theorem result (name : String) (implementation : Implements program name)
    (limit : Int) (base : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program name [integer limit] base
      (.ref (base.length + 1)) finalHeap ∧
      finalHeap[base.length + 1]? = some
        (.list (entries (limit + 1).toNat (fun n => number n.totient))) := by
  obtain ⟨body, found, executed⟩ := implementation
  by_cases small : limit < 1
  · let entry := HeapCalls.argumentEnv sieveArgs [integer limit]
    let phiEnv := setLocal entry "phi" (.ref (base.length + 1))
    refine ⟨phiHeap base (limit + 1).toNat, ?_, ?_⟩
    · apply BlockReturns.function found rfl
      rw [executed]
      apply BlockReturns.assign (allocate_phi (program := program) limit base entry rfl)
      apply BlockReturns.branch (flag := true)
        (by simpa only [small, decide_true] using
          small_result (program := program) limit (phiHeap base (limit + 1).toNat) phiEnv rfl) rfl
      exact BlockReturns.return_value (Evaluates.lookup rfl)
    · have cells : entries (limit + 1).toNat (fun _ => number 0) =
          entries (limit + 1).toNat (fun n => number n.totient) := by
        apply List.map_congr_left
        intro n member
        have bounded := List.mem_range.mp member
        have zero : n = 0 := by omega
        subst n
        rfl
      have found : (phiHeap base (limit + 1).toNat)[base.length + 1]? =
          some (.list (entries (limit + 1).toNat (fun _ => number 0))) := by simp [phiHeap]
      exact found.trans (congrArg (fun items => some (Mutable.Object.list items)) cells)
  · have cast : (limit.toNat : Int) = limit := Int.toNat_of_nonneg (by omega)
    obtain ⟨finalCache, returned⟩ := positive_result (program := program)
      name ⟨body, found, executed⟩ limit.toNat (by omega) base
    refine ⟨realize base (limit.toNat + 1) (expected (limit.toNat + 1)) finalCache, ?_, ?_⟩
    · simpa only [integer, number, cast] using returned
    · have count : (limit + 1).toNat = limit.toNat + 1 := by omega
      have cells : entries (limit.toNat + 1) (fun n => number (expected (limit.toNat + 1) n).1) =
          entries (limit.toNat + 1) (fun n => number n.totient) := by
        apply List.map_congr_left
        intro n member
        rw [expected_finished limit.toNat n (by have h := List.mem_range.mp member; omega)]
      simpa only [count, cells] using phi_found base (limit.toNat + 1)
        (fun n => (expected (limit.toNat + 1) n).1)
        (fun n => (expected (limit.toNat + 1) n).2) finalCache

end ProjectEulerSolutions.TotientSieveCalls
