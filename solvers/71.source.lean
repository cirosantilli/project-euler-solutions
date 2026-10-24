import «.formalsw».solvers.«71.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.P71

/-! Source execution of the bounded denominator scan, including both continue transfers. -/
namespace solvers.«71»
open FormalSW.PyAst HeapCalls ProjectEulerStatements.FractionSearch
set_option Elab.async false

abbrev program := Solvers.P71Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)
abbrev pairObject (p : Pair) : Mutable.Object := .tuple [number p.1, number p.2]

structure Ready (a b : Nat) (best : Pair) (env : HeapCalls.Env) : Prop where
  num : env "target_n" = some (number a)
  den : env "target_d" = some (number b)
  bestN : env "best_n" = some (number best.1)
  bestD : env "best_d" = some (number best.2)

abbrev numeratorExpression : Expr :=
  .binOp (.binOp (.binOp (.name "target_n") .mult (.name "d")) .sub (.constant (.int 1)))
    .floorDiv (.name "target_d")

abbrev zeroCondition : Expr := .compare (.name "n") [(.ltE, .constant (.int 0))]
abbrev gcdCondition : Expr := .compare (.call (.name "gcd") [.name "n", .name "d"])
  [(.notEq, .constant (.int 1))]
abbrev improvement : Expr := .compare (.binOp (.name "n") .mult (.name "best_d"))
  [(.gt, .binOp (.name "best_n") .mult (.name "d"))]
abbrev iteration : List Stmt :=
  [.assign [.name "n"] numeratorExpression,
   .ifThen zeroCondition [.continueLoop] [],
   .ifThen gcdCondition [.continueLoop] [],
   .ifThen improvement
     [.assign [.tuple [.name "best_n", .name "best_d"]] (.tuple [.name "n", .name "d"])] []]
abbrev finish : Stmt := .return (.tuple [.name "best_n", .name "best_d"])

theorem numerator_result (a b d : Nat) (best : Pair) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (ready : Ready a b best env) (getD : env "d" = some (number d))
    (positive : 0 < a) (denPositive : 0 < b) (dPositive : 0 < d) :
    Evaluates program env numeratorExpression heap (number (ceiling a b d)) heap := by
  have productPositive : 1 ≤ a * d := Nat.mul_pos positive dPositive
  have castSub : ((a * d - 1 : Nat) : Int) = (a : Int) * d - 1 := by
    rw [Int.natCast_sub productPositive]
    simp
  apply Evaluates.binary (av := .scalar (.int ((a : Int) * d - 1))) (bv := number b)
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (av := number (a * d)) (bv := number 1)
      (middle := heap) (afterRight := heap)
      (Evaluates.binary (middle := heap) (afterRight := heap)
        (Evaluates.lookup ready.num) (Evaluates.lookup getD)
        (by simp [Mutable.binary, scalarBinary])) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary])
  · exact Evaluates.lookup ready.den
  · simp only [Mutable.binary, scalarBinary, number, ceiling]
    simp [castSub, Int.natCast_ediv, denPositive.ne', denPositive]

theorem zero_result (n : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) :
    Evaluates program env zeroCondition heap (flag (decide (n = 0))) heap := by
  apply Evaluates.compare (Evaluates.lookup getN) Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem gcd_result (n d : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getD : env "d" = some (number d)) :
    Evaluates program env gcdCondition heap (flag (decide (¬ Nat.Coprime n d))) heap := by
  apply Evaluates.compare (av := number (Nat.gcd n d)) (bv := number 1)
    (middle := heap) (afterRight := heap)
  · apply Evaluates.library_call
      (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getD) .nil)) rfl
    change HeapCalls.scalarResult (some (Math.gcdCall [.int n, .int d])) heap = _
    simp [Math.gcdCall_pair, HeapCalls.scalarResult]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, Nat.Coprime]

theorem improvement_result (n d : Nat) (best : Pair) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getN : env "n" = some (number n)) (getD : env "d" = some (number d))
    (getBestN : env "best_n" = some (number best.1))
    (getBestD : env "best_d" = some (number best.2)) :
    Evaluates program env improvement heap (flag (decide (best.1 * d < n * best.2))) heap := by
  apply Evaluates.compare (av := number (n * best.2)) (bv := number (best.1 * d))
    (middle := heap) (afterRight := heap)
  · exact Evaluates.binary (middle := heap) (afterRight := heap)
      (Evaluates.lookup getN) (Evaluates.lookup getBestD) (by simp [Mutable.binary, scalarBinary])
  · exact Evaluates.binary (middle := heap) (afterRight := heap)
      (Evaluates.lookup getBestN) (Evaluates.lookup getD) (by simp [Mutable.binary, scalarBinary])
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    norm_cast

/-- The final tuple is fresh; earlier unpacking allocations can be arbitrary garbage. -/
theorem loop_result (a b : Nat) (stop : Int) (first count : Nat) (best : Pair)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (positive : 0 < a) (denPositive : 0 < b)
    (firstPositive : 0 < first) (length : (stop - first).toNat = count)
    (ready : Ready a b best env) :
    ∃ before, BlockReturns program env
      [.rangeStepLoop "d" first stop 1 iteration [], finish]
      heap (.ref before.length)
      (before ++ [pairObject
        ((List.range' first count).foldl (fun p d => optimizedStep a b d p) best)]) := by
  induction count generalizing first best env heap with
  | zero =>
      refine ⟨heap, ?_⟩
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.tuple
        (.cons (Evaluates.lookup ready.bestN)
          (.cons (Evaluates.lookup ready.bestD) .nil)) (by simp))
  | succ count ih =>
      let entered := setLocal env "d" (number first)
      let n := ceiling a b first
      let nextEnv := setLocal entered "n" (number n)
      have enteredReady : Ready a b best entered := by
        constructor <;> simp [entered, setLocal, ready.num, ready.den, ready.bestN, ready.bestD]
      have nextReady : Ready a b best nextEnv := by
        constructor <;> simp [nextEnv, setLocal, enteredReady.num, enteredReady.den,
          enteredReady.bestN, enteredReady.bestD]
      have getD : nextEnv "d" = some (number first) := by simp [nextEnv, entered, setLocal]
      have getN : nextEnv "n" = some (number n) := by simp [nextEnv, setLocal]
      have testedZero := zero_result n nextEnv heap getN
      have evaluated := numerator_result a b first best entered heap enteredReady
        (by simp [entered, setLocal]) positive denPositive firstPositive
      have nextLength : (stop - (first + 1)).toNat = count := by omega
      by_cases zero : n = 0
      · have step : optimizedStep a b first best = best := by
          simp only [optimizedStep, show ceiling a b first = n from rfl, if_pos zero]
        obtain ⟨before, rest⟩ := ih (first + 1) best nextEnv heap (by omega) nextLength nextReady
        refine ⟨before, ?_⟩
        apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
        apply BlockReturns.assign evaluated
        apply BlockReturns.branch (flag := true)
          (by simpa only [zero, decide_true] using testedZero) rfl
        apply BlockReturns.continue_loop rfl
        simpa only [List.range'_succ, List.foldl_cons, step,
          Int.natCast_add, Int.natCast_one] using rest
      · have testedGcd := gcd_result n first nextEnv heap getN getD
        by_cases coprime : Nat.Coprime n first
        · have testedBetter := improvement_result n first best nextEnv heap getN getD
            nextReady.bestN nextReady.bestD
          by_cases better : best.1 * first < n * best.2
          · have step : optimizedStep a b first best = (n, first) := by
              simp only [optimizedStep, show ceiling a b first = n from rfl, if_neg zero,
                if_pos coprime, choose, if_pos better]
            let items := [number n, number first]
            let changed := assignNames nextEnv ["best_n", "best_d"] items
            have changedReady : Ready a b (n, first) changed := by
              constructor <;> simp [changed, items, assignNames, setLocal, nextReady.num,
                nextReady.den]
            obtain ⟨before, rest⟩ := ih (first + 1) (n, first) changed
              (heap ++ [pairObject (n, first)])
              (by omega) nextLength changedReady
            refine ⟨before, ?_⟩
            apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
            apply BlockReturns.assign evaluated
            apply BlockReturns.branch (flag := false)
              (by simpa only [zero, decide_false] using testedZero) rfl
            apply BlockReturns.branch (flag := false)
              (by simpa [Nat.Coprime, coprime] using testedGcd) rfl
            apply BlockReturns.branch (flag := true)
              (by simpa only [better, decide_true] using testedBetter) rfl
            apply BlockReturns.unpack (names := ["best_n", "best_d"])
              (items := items) (middle := heap ++ [pairObject (n, first)])
              (afterUnpack := heap ++ [pairObject (n, first)]) rfl
              (Evaluates.tuple (.cons (Evaluates.lookup getN)
                (.cons (Evaluates.lookup getD) .nil)) (by simp))
              (by simp [Mutable.sequence, pairObject, items]) rfl
            apply BlockReturns.loop_resume
            simpa only [List.range'_succ, List.foldl_cons, step,
              Int.natCast_add, Int.natCast_one] using rest
          · have step : optimizedStep a b first best = best := by
              simp only [optimizedStep, show ceiling a b first = n from rfl, if_neg zero,
                if_pos coprime, choose, if_neg better]
            obtain ⟨before, rest⟩ := ih (first + 1) best nextEnv heap
              (by omega) nextLength nextReady
            refine ⟨before, ?_⟩
            apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
            apply BlockReturns.assign evaluated
            apply BlockReturns.branch (flag := false)
              (by simpa only [zero, decide_false] using testedZero) rfl
            apply BlockReturns.branch (flag := false)
              (by simpa [Nat.Coprime, coprime] using testedGcd) rfl
            apply BlockReturns.branch (flag := false)
              (by simpa only [better, decide_false] using testedBetter) rfl
            apply BlockReturns.loop_resume
            simpa only [List.range'_succ, List.foldl_cons, step,
              Int.natCast_add, Int.natCast_one] using rest
        · have step : optimizedStep a b first best = best := by
            simp only [optimizedStep, show ceiling a b first = n from rfl,
              if_neg zero, if_neg coprime]
          obtain ⟨before, rest⟩ := ih (first + 1) best nextEnv heap (by omega) nextLength nextReady
          refine ⟨before, ?_⟩
          apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
          apply BlockReturns.assign evaluated
          apply BlockReturns.branch (flag := false)
            (by simpa only [zero, decide_false] using testedZero) rfl
          apply BlockReturns.branch (flag := true)
            (by simpa [Nat.Coprime, coprime] using testedGcd) rfl
          apply BlockReturns.continue_loop rfl
          simpa only [List.range'_succ, List.foldl_cons, step,
            Int.natCast_add, Int.natCast_one] using rest

namespace best_fraction_left_of

theorem result (a b : Nat) (limit : Int) (heap : Mutable.Heap)
    (positive : 0 < a) (denPositive : 0 < b) :
    ∃ before, ReturnsValue program "best_fraction_left_of"
      [number a, number b, .scalar (.int limit)] heap (.ref before.length)
      (before ++ [pairObject (optimizedUpTo a b limit.toNat)]) := by
  let initial := heap ++ [pairObject (0, 1)]
  let env := HeapCalls.argumentEnv
    ((findFunctionIn program.body "best_fraction_left_of").getD ([], [])).1
    [number a, number b, .scalar (.int limit)]
  let ready := assignNames env ["best_n", "best_d"] [number 0, number 1]
  obtain ⟨before, rest⟩ := loop_result a b (limit + 1) 1 limit.toNat (0, 1) ready initial
    positive denPositive (by decide) (by omega) ⟨rfl, rfl, rfl, rfl⟩
  refine ⟨before, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.unpack (names := ["best_n", "best_d"]) (items := [number 0, number 1])
    (middle := initial) (afterUnpack := initial) rfl
    (Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp))
    (by simp [initial, Mutable.sequence, pairObject]) rfl
  apply BlockReturns.for_range (middle := initial)
    (values := [number 1, .scalar (.int (limit + 1))])
    (.cons Evaluates.constant (.cons (Evaluates.binary
      (Evaluates.lookup rfl) Evaluates.constant (by simp [Mutable.binary, scalarBinary])) .nil)) rfl
  have folded : (List.range' 1 limit.toNat).foldl (fun p d => optimizedStep a b d p) (0, 1) =
      optimizedUpTo a b limit.toNat := by
    simpa only [Nat.zero_add] using optimized_fold a b 0 limit.toNat
  simpa only [folded] using rest

end best_fraction_left_of

namespace solve

theorem body_result (limit : Int) (a b : Nat) (heap : Mutable.Heap)
    (positive : 0 < a) (denPositive : 0 < b) :
    ∃ finalHeap, BlockReturns program
      (HeapCalls.argumentEnv ((findFunctionIn program.body "solve").getD ([], [])).1
        [.scalar (.int limit), number a, number b])
      (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)
      heap (number (optimizedUpTo a b limit.toNat).1) finalHeap := by
  obtain ⟨before, returned⟩ := best_fraction_left_of.result a b limit heap positive denPositive
  refine ⟨before ++ [pairObject (optimizedUpTo a b limit.toNat)], ?_⟩
  apply BlockReturns.return_value
  apply Evaluates.subscript (middle := before ++ [pairObject (optimizedUpTo a b limit.toNat)])
    (afterIndex := before ++ [pairObject (optimizedUpTo a b limit.toNat)])
    (value := .ref before.length) (position := number 0) (by intros; intro h; cases h)
  · exact Evaluates.call (.cons (Evaluates.lookup rfl)
      (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil))) returned
  · exact Evaluates.constant
  · simp [Mutable.getItem, Mutable.index, pairObject, Except.map]

theorem result (limit num den : Int) (heap : Mutable.Heap)
    (positive : 0 < num) (proper : num < den) :
    ∃ finalHeap, ReturnsValue program "solve"
      [.scalar (.int limit), .scalar (.int num), .scalar (.int den)]
      heap (number (ProjectEulerSolutions.P71.solve limit num den)) finalHeap := by
  obtain ⟨finalHeap, body⟩ := body_result limit num.toNat den.toNat heap (by omega) (by omega)
  have castN : (num.toNat : Int) = num := Int.toNat_of_nonneg (by omega)
  have castD : (den.toNat : Int) = den := Int.toNat_of_nonneg (by omega)
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  simpa only [number, castN, castD, ProjectEulerSolutions.P71.solve,
    ProjectEulerSolutions.P71.bestFractionLeftOf_correct] using body

theorem one_argument_result (limit : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [.scalar (.int limit)]
      heap (number (ProjectEulerSolutions.P71.solve limit)) finalHeap := by
  obtain ⟨finalHeap, returned⟩ := result limit 3 7 heap (by decide) (by decide)
  exact ⟨finalHeap, ReturnsValue.with_defaults rfl rfl rfl returned⟩

theorem two_argument_result (limit num : Int) (heap : Mutable.Heap)
    (positive : 0 < num) (proper : num < 7) :
    ∃ finalHeap, ReturnsValue program "solve" [.scalar (.int limit), .scalar (.int num)]
      heap (number (ProjectEulerSolutions.P71.solve limit num)) finalHeap := by
  obtain ⟨finalHeap, returned⟩ := result limit num 7 heap positive proper
  exact ⟨finalHeap, ReturnsValue.with_defaults rfl rfl rfl returned⟩

end solve
end solvers.«71»
