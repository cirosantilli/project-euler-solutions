import «.formalsw».solvers.«44.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p44.py»
import ProjectEulerSolutions.Lib.PentagonalDiscriminant
import ProjectEulerSolutions.Lib.PentagonalSearch
import lang.py.FormalSW.LoopCalls

set_option Elab.async false

namespace solvers.«44»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open ProjectEulerSolutions.PentagonalDiscriminant
open ProjectEulerSolutions.PentagonalSearch
open data.«project-euler-statements».data.py.project_euler_statements.p44 (polynomial polynomial_nat pentagonalTest)
private abbrev program := Solvers.P44Ast.module
private abbrev number (n : Int) : Mutable.Value := .scalar (.int n)
private abbrev natural (n : Nat) : Mutable.Value := number n
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

namespace pentagonal
/-- The optimized table stores exactly the source's integer pentagonal polynomial. -/
theorem result (n : Int) (heap : Mutable.Heap) :
    ReturnsValue program "pentagonal" [number n] heap (number (polynomial n)) heap := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.binary (av := number (n*(3*n-1))) (bv := number 2) (middle := heap) (afterRight := heap)
  · apply Evaluates.binary (av := number n) (bv := number (3*n-1)) (middle := heap) (afterRight := heap)
      (Evaluates.lookup rfl)
    · exact Evaluates.binary (av := number (3*n)) (bv := number 1) (middle := heap) (afterRight := heap)
        (Evaluates.binary (av := number 3) (bv := number n) (middle := heap) (afterRight := heap)
          Evaluates.constant (Evaluates.lookup rfl) rfl) Evaluates.constant rfl
    · rfl
  · exact Evaluates.constant
  · rfl
end pentagonal

namespace is_pentagonal
private def squareTest : Expr := .compare (.binOp (.name "s") .mult (.name "s")) [(.eq, .name "t")]
private def remainderTest : Expr := .compare
  (.binOp (.binOp (.constant (.int 1)) .add (.name "s")) .mod (.constant (.int 6))) [(.eq, .constant (.int 0))]

/-- The exact integer-square-root criterion agrees with the exhaustive positive-index predicate. -/
theorem result (x : Int) (heap : Mutable.Heap) :
    ReturnsValue program "is_pentagonal" [number x] heap (flag (pentagonalTest x)) heap := by
  by_cases nonpositive : x ≤ 0
  · have falseTest : pentagonalTest x = false := by
      by_cases negative : x < 0
      · simp [pentagonalTest, negative]
      · have zero : x=0 := by omega
        subst x
        rfl
    rw [falseTest]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (value := flag true) (flag := true)
      (Evaluates.compare (av := number x) (bv := number 0) (middle := heap) (afterRight := heap) (finalHeap := heap)
        (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, nonpositive])) rfl
    exact BlockReturns.return_value Evaluates.constant
  · lift x to Nat using (by omega : 0 ≤ x)
    have positive : 0 < x := by omega
    let initial := functionEnv ((findFunctionIn program.body "is_pentagonal").getD ([], [])).1
      ((findFunctionIn program.body "is_pentagonal").getD ([], [])).2 [number x] []
    let d := 24*x+1
    let root := Nat.sqrt d
    let discriminated := setLocal initial "t" (number d)
    let rooted := setLocal discriminated "s" (number root)
    have getT : discriminated "t" = some (number d) := by simp [discriminated, setLocal]
    have setT : Evaluates program initial
        (.binOp (.binOp (.constant (.int 24)) .mult (.name "x")) .add (.constant (.int 1))) heap (number d) heap := by
      apply Evaluates.binary (av := natural (24*x)) (bv := number 1) (middle := heap) (afterRight := heap)
        (Evaluates.binary Evaluates.constant (Evaluates.lookup rfl) (by simp [Mutable.binary, scalarBinary])) Evaluates.constant
      simp [Mutable.binary, scalarBinary, number, d]
    have rootResult : Evaluates program discriminated (.call (.name "isqrt") [.name "t"]) heap (number root) heap := by
      apply Evaluates.library_call (.cons (Evaluates.lookup getT) .nil) rfl
      have same : Math.isqrt d = Nat.sqrt d := Nat.eq_sqrt.mpr (Math.isqrt_spec d)
      change scalarResult (some (Math.isqrtCall [.int d])) heap = _
      simp [scalarResult, Math.isqrtCall, same, root, number, show ¬ (d : Int) < 0 by omega]
    have squared : Evaluates program rooted squareTest heap (flag (decide (root*root=d))) heap := by
      apply Evaluates.compare (av := natural (root*root)) (bv := number d) (middle := heap) (afterRight := heap)
        (Evaluates.binary (av := number root) (bv := number root) (middle := heap) (afterRight := heap) (finalHeap := heap)
          (Evaluates.lookup (by simp [rooted, setLocal])) (Evaluates.lookup (by simp [rooted, setLocal]))
          (by simp [Mutable.binary, scalarBinary]))
        (Evaluates.lookup (by simp [rooted, discriminated, setLocal]))
      change some (Except.ok (ε := ExceptionObject) (flag (decide (((root*root : Nat) : Int) = (d : Int)))), heap) = _
      simp only [Int.natCast_inj]
    have remainder : Evaluates program rooted remainderTest heap (flag (decide ((1+root)%6=0))) heap := by
      apply Evaluates.compare (av := natural ((1+root)%6)) (bv := number 0) (middle := heap) (afterRight := heap)
      · apply Evaluates.binary (av := natural (1+root)) (bv := number 6) (middle := heap) (afterRight := heap)
          (Evaluates.binary (av := number 1) (bv := number root) (middle := heap) (afterRight := heap) (finalHeap := heap)
            Evaluates.constant (Evaluates.lookup (by simp [rooted, setLocal]))
            (by simp [Mutable.binary, scalarBinary])) Evaluates.constant
        simp [Mutable.binary, scalarBinary]
      · exact Evaluates.constant
      · change some (Except.ok (ε := ExceptionObject) (flag (decide ((((1+root)%6 : Nat) : Int) = 0))), heap) = _
        simp only [Int.natCast_eq_zero]
    have final : Evaluates program rooted (.boolOp .and [squareTest, remainderTest]) heap (flag (discriminant x)) heap := by
      by_cases square : root*root=d
      · simp only [discriminant, if_neg (by omega : x ≠ 0), show Nat.sqrt (24*x+1)*Nat.sqrt (24*x+1)=24*x+1 from square, true_and]
        exact Evaluates.boolean_next (flag := true) (by simpa only [square, decide_true] using squared) rfl rfl
          (Evaluates.boolean_last remainder rfl)
      · simp only [discriminant, if_neg (by omega : x ≠ 0), show ¬ Nat.sqrt (24*x+1)*Nat.sqrt (24*x+1)=24*x+1 from square, false_and, decide_false]
        exact Evaluates.boolean_stop (flag := false) (by simpa only [square, decide_false] using squared) rfl rfl
    rw [pentagonalTest, if_neg (by omega : ¬ (x : Int) < 0), Int.toNat_natCast, ← correct]
    apply BlockReturns.function rfl rfl
    apply BlockReturns.branch (value := flag false) (flag := false)
      (Evaluates.compare (av := number x) (bv := number 0) (middle := heap) (afterRight := heap) (finalHeap := heap)
        (Evaluates.lookup rfl) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp]; omega)) rfl
    apply BlockReturns.assign setT
    apply BlockReturns.assign rootResult
    exact BlockReturns.return_value final
end is_pentagonal

namespace Table
private def entries (first count : Nat) : List Mutable.Value :=
  ((List.range first).map fun i => natural (pentagonal i)) ++ List.replicate count (natural 0)
def heap (base : Mutable.Heap) (first count : Nat) : Mutable.Heap :=
  ProjectEulerSolutions.HeapCollection.working base (entries first count) []

private theorem entries_length (first count : Nat) : (entries first count).length = first+count := by
  simp [entries]

private theorem entries_set (first count : Nat) :
    (entries first (count+1)).set first (natural (pentagonal first)) = entries (first+1) count := by
  simp [entries, List.replicate_succ, List.range_succ, List.map_append, List.set_append_right, List.append_assoc]

private theorem found (base : Mutable.Heap) (first count : Nat) :
    (heap base first count)[base.length]? = some (.list (entries first count)) :=
  ProjectEulerSolutions.HeapCollection.found _ _ _

private theorem write (heap : Mutable.Heap) (address index : Nat) (items : List Mutable.Value) (value : Mutable.Value)
    (found : heap[address]? = some (.list items)) (inside : index < items.length) :
    Mutable.setItem (.ref address) (natural index) value heap =
      some (.ok (.scalar .none), heap.set address (.list (items.set index value))) := by
  have positive : ¬ (index : Int) < 0 := by omega
  have bounded : ¬ items.length ≤ index := by omega
  simp [Mutable.setItem, found, Mutable.index, positive, bounded, Except.map]

private def rowBody : List Stmt := [.assign [.subscript (.name "pent") (.name "n")] (.call (.name "pentagonal") [.name "n"])]

/-- Each source table write replaces exactly the next zero slot with its pentagonal value. -/
theorem range_result (first count stop : Nat) (equation : first+count=stop) (base : Mutable.Heap) (env : HeapCalls.Env)
    (getTable : env "pent" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "n" first stop 1 rowBody []]
        (heap base first count) finalEnv (heap base stop 0) ∧
      (∀ name, name ≠ "n" → finalEnv name = env name) := by
  induction count generalizing first env with
  | zero =>
      have equal : first=stop := by omega
      subst first
      refine ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, ?_⟩
      intros; rfl
  | succ count ih =>
      let bound := setLocal env "n" (natural first)
      have step : BlockContinues program bound rowBody (heap base first (count+1)) bound (heap base (first+1) count) := by
        apply BlockContinues.setItem (value := natural (pentagonal first)) (afterRhs := heap base first (count+1))
          (afterTarget := heap base first (count+1)) (receiver := .ref base.length) (subscript := natural first)
          (by intros; simp)
          (Evaluates.call (values := [natural first]) (.cons (Evaluates.lookup (by simp [bound, setLocal])) .nil)
            (by simpa only [polynomial_nat] using pentagonal.result (first : Int) (heap base first (count+1))))
          (.cons (Evaluates.lookup (by simpa [bound, setLocal] using getTable)) (.cons (Evaluates.lookup (by simp [bound, setLocal])) .nil))
        have written := write (heap base first (count+1)) base.length first (entries first (count+1)) (natural (pentagonal first))
          (found base first (count+1)) (by rw [entries_length]; omega)
        simpa only [heap, ProjectEulerSolutions.HeapCollection.store, entries_set] using written
      obtain ⟨finalEnv, executed, untouched⟩ := ih (first+1) (by omega) bound (by simpa [bound, setLocal] using getTable)
      refine ⟨finalEnv, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, show Calls.loopControl rowBody = false from rfl,
          Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using step.append executed
      · intro name unrelated
        rw [untouched name unrelated]
        simp [bound, setLocal, unrelated]

private theorem initialize_result (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (natural limit)) :
    Evaluates program env (.binOp (.list [.constant (.int 0)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) heap (.ref (heap.length+1))
      (Table.heap (heap ++ [.list [natural 0]]) 1 limit) := by
  let base := heap ++ [.list [natural 0]]
  apply Evaluates.binary (av := .ref heap.length) (bv := natural (limit+1)) (middle := base) (afterRight := base)
  · exact Evaluates.list (.cons Evaluates.constant .nil)
  · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
  · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl, Mutable.array, Mutable.allocate,
      base, Table.heap, entries, ProjectEulerSolutions.HeapCollection.working, pentagonal, Except.map, List.replicate_succ]

def sourceSetup : List Stmt := (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).take 2

/-- The source list multiplication and fill loop produce every table entry, including P(0). -/
theorem result (limit : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getLimit : env "limit" = some (natural limit)) :
    ∃ finalEnv, BlockContinues program env sourceSetup heap finalEnv
        (Table.heap (heap ++ [.list [natural 0]]) (limit+1) 0) ∧
      finalEnv "pent" = some (.ref (heap.length+1)) ∧
      (∀ name, name ∉ ["pent", "n"] → finalEnv name = env name) := by
  let base := heap ++ [.list [natural 0]]
  let ready := setLocal env "pent" (.ref base.length)
  obtain ⟨finalEnv, executed, untouched⟩ := range_result 1 limit (limit+1) (by omega) base ready (by simp [ready, setLocal])
  have loop : BlockContinues program ready
      [.forIn (.name "n") (.call (.name "range") [.constant (.int 1), .binOp (.name "limit") .add (.constant (.int 1))]) rowBody []]
      (Table.heap base 1 limit) finalEnv (Table.heap base (limit+1) 0) := by
    exact BlockContinues.for_range (values := [natural 1, natural (limit+1)])
      (.cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup (by simpa [ready, setLocal] using getLimit)) Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) .nil)) (by simp [rangeBounds]; rfl) executed
  refine ⟨finalEnv, ?_, ?_, ?_⟩
  · have allocation : BlockContinues program env (sourceSetup.take 1) heap ready (Table.heap base 1 limit) := by
      simpa only [ready, base, List.length_append, List.length_singleton] using
        (BlockContinues.annAssign (annotation := .subscript (.name "list") (.name "int")) (name := "pent") (initialize_result limit env heap getLimit))
    exact allocation.append loop
  · rw [untouched "pent" (by decide)]
    simp [ready, setLocal, base]
  · intro name unrelated
    have np : name ≠ "pent" := by intro equal; subst name; simp at unrelated
    have nn : name ≠ "n" := by intro equal; subst name; simp at unrelated
    rw [untouched name nn]
    simp [ready, setLocal, np]

/-- A filled table lookup returns the pentagonal number at its original list address. -/
theorem lookup (heap : Mutable.Heap) (address limit index : Nat) (bound : index ≤ limit)
    (found : heap[address]? = some (.list (entries (limit+1) 0))) :
    Mutable.getItem (.ref address) (natural index) heap = some (.ok (natural (pentagonal index)), heap) := by
  have inside : index < (entries (limit+1) 0).length := by rw [entries_length]; omega
  have value : (entries (limit+1) 0)[index] = natural (pentagonal index) := by
    simp [entries, List.getElem_map, List.getElem_range]
  simpa only [value] using Mutable.getItem_list heap address index (entries (limit+1) 0) found inside

end Table

namespace Search
private def smaller : Expr := .subscript (.name "pent") (.name "j")
private def differenceExpression : Expr := .binOp (.name "pk") .sub smaller
private def sumExpression : Expr := .binOp (.name "pk") .add smaller
private def cutTest : Expr := .compare (.name "diff") [(.gtE, .name "best")]
private def predicate : Expr := .boolOp .and
  [.call (.name "is_pentagonal") [.name "diff"], .call (.name "is_pentagonal") [sumExpression]]
private def updateBest : Stmt := .ifThen predicate [.assign [.name "best"] (.name "diff")] []
def innerBody : List Stmt := [.assign [.name "diff"] differenceExpression, .ifThen cutTest [.breakLoop] [], updateBest]
def sourceInnerBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3 with
  | .forIn _ _ (_ :: .forIn _ _ body _ :: _) _ :: _ => body
  | _ => []

theorem innerBody_source : innerBody = sourceInnerBody := rfl

private theorem read_smaller (limit j : Nat) (within : j ≤ limit) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (Table.entries (limit+1) 0))) (env : HeapCalls.Env)
    (getTable : env "pent" = some (.ref address)) (getJ : env "j" = some (natural j)) :
    Evaluates program env smaller heap (natural (pentagonal j)) heap :=
  Evaluates.subscript (by intros; simp) (Evaluates.lookup getTable) (Evaluates.lookup getJ) (Table.lookup heap address limit j within found)

private theorem difference_result (limit k j : Nat) (within : j ≤ limit) (less : j ≤ k)
    (heap : Mutable.Heap) (address : Nat) (found : heap[address]? = some (.list (Table.entries (limit+1) 0)))
    (env : HeapCalls.Env) (getTable : env "pent" = some (.ref address)) (getJ : env "j" = some (natural j))
    (getPK : env "pk" = some (natural (pentagonal k))) :
    Evaluates program env differenceExpression heap (natural (difference k j)) heap :=
  Evaluates.binary (Evaluates.lookup getPK) (read_smaller limit j within heap address found env getTable getJ)
    (by simp [Mutable.binary, scalarBinary, difference, Int.natCast_sub (strictMono.monotone less)])

private theorem predicate_result (limit k j : Nat) (within : j ≤ limit)
    (heap : Mutable.Heap) (address : Nat) (found : heap[address]? = some (.list (Table.entries (limit+1) 0)))
    (env : HeapCalls.Env) (getTable : env "pent" = some (.ref address)) (getJ : env "j" = some (natural j))
    (getPK : env "pk" = some (natural (pentagonal k))) (getDiff : env "diff" = some (natural (difference k j))) :
    Evaluates program env predicate heap (flag (accepted k j)) heap := by
  have diffTest : Evaluates program env (.call (.name "is_pentagonal") [.name "diff"]) heap
      (flag (exhaustive (difference k j))) heap := by
    apply Evaluates.call (.cons (Evaluates.lookup getDiff) .nil)
    simpa only [pentagonalTest, if_neg (Int.not_lt.mpr (Int.natCast_nonneg _)), Int.toNat_natCast] using
      is_pentagonal.result ((difference k j : Nat) : Int) heap
  have sumResult : Evaluates program env sumExpression heap (natural (pentagonal k + pentagonal j)) heap :=
    Evaluates.binary (Evaluates.lookup getPK) (read_smaller limit j within heap address found env getTable getJ)
      (by simp [Mutable.binary, scalarBinary])
  have sumTest : Evaluates program env (.call (.name "is_pentagonal") [sumExpression]) heap
      (flag (exhaustive (pentagonal k + pentagonal j))) heap := by
    apply Evaluates.call (.cons sumResult .nil)
    simpa only [pentagonalTest, if_neg (Int.not_lt.mpr (Int.natCast_nonneg _)), Int.toNat_natCast] using
      is_pentagonal.result ((pentagonal k + pentagonal j : Nat) : Int) heap
  cases keep : exhaustive (difference k j) with
  | false =>
      simp only [accepted, keep, Bool.and_false]
      exact Evaluates.boolean_stop (flag := false) (by simpa only [keep] using diffTest) rfl rfl
  | true =>
      simp only [accepted, keep, Bool.and_true]
      exact Evaluates.boolean_next (flag := true) (by simpa only [keep] using diffTest) rfl rfl
        (Evaluates.boolean_last sumTest rfl)

/-- The descending source row either breaks at a proven lower bound or visits the next smaller index. -/
theorem inner_result (limit k count best : Nat) (within : k ≤ limit) (less : count < k)
    (heap : Mutable.Heap) (address : Nat) (found : heap[address]? = some (.list (Table.entries (limit+1) 0)))
    (env : HeapCalls.Env) (getTable : env "pent" = some (.ref address))
    (getPK : env "pk" = some (natural (pentagonal k))) (getBest : env "best" = some (natural best)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "j" count 0 (-1) innerBody []] heap finalEnv heap ∧
      finalEnv "best" = some (natural (row k count best)) ∧
      (∀ name, name ∉ ["j", "diff", "best"] → finalEnv name = env name) := by
  induction count generalizing best env with
  | zero =>
      refine ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, getBest, ?_⟩
      intros; rfl
  | succ count ih =>
      let bound := setLocal env "j" (natural (count+1))
      let diffed := setLocal bound "diff" (natural (difference k (count+1)))
      have assigned : BlockContinues program bound [.assign [.name "diff"] differenceExpression] heap diffed heap :=
        BlockContinues.assign (difference_result limit k (count+1) (by omega) (by omega) heap address found bound
          (by simpa [bound, setLocal] using getTable) (by simp [bound, setLocal]) (by simpa [bound, setLocal] using getPK))
      have getDiff : diffed "diff" = some (natural (difference k (count+1))) := by simp [diffed, setLocal]
      have getCurrentBest : diffed "best" = some (natural best) := by simpa [diffed, bound, setLocal] using getBest
      have tested : Evaluates program diffed cutTest heap (flag (decide (best ≤ difference k (count+1)))) heap :=
        Evaluates.compare (Evaluates.lookup getDiff) (Evaluates.lookup getCurrentBest)
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
      have next : ((count+1 : Nat) : Int)+(-1)=(count : Int) := by omega
      by_cases cut : best ≤ difference k (count+1)
      · refine ⟨diffed, ?_, ?_, ?_⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues])
          simp only [Calls.loopContinuation, show Calls.loopControl innerBody = true from rfl, if_true, next]
          apply assigned.append
          apply BlockContinues.branch_next (flag := true) tested (by simp [Mutable.truth, scalarTruth, cut])
          exact BlockContinues.break_loop rfl .nil
        · rw [prune k (count+1) best cut]
          exact getCurrentBest
        · intro name unrelated
          have nj : name ≠ "j" := by intro equal; subst name; simp at unrelated
          have nd : name ≠ "diff" := by intro equal; subst name; simp at unrelated
          simp [diffed, bound, setLocal, nj, nd]
      · have better : difference k (count+1) ≤ best := by omega
        let newBest := if accepted k (count+1) then difference k (count+1) else best
        let ready := if accepted k (count+1) then setLocal diffed "best" (natural (difference k (count+1))) else diffed
        have acceptedTest := predicate_result limit k (count+1) (by omega) heap address found diffed
          (by simpa [diffed, bound, setLocal] using getTable) (by simp [diffed, bound, setLocal])
          (by simpa [diffed, bound, setLocal] using getPK) getDiff
        have updated : BlockContinues program diffed [updateBest] heap ready heap := by
          cases keep : accepted k (count+1) with
          | false =>
              simp only [ready, keep, Bool.false_eq_true, if_false]
              exact BlockContinues.branch (flag := false) acceptedTest (by simp [Mutable.truth, scalarTruth, keep]) .nil
          | true =>
              simp only [ready, keep, if_true]
              exact BlockContinues.branch (flag := true) acceptedTest (by simp [Mutable.truth, scalarTruth, keep])
                (BlockContinues.assign (Evaluates.lookup getDiff))
        have readyBest : ready "best" = some (natural newBest) := by
          cases keep : accepted k (count+1) <;> simp [ready, newBest, keep, setLocal, getCurrentBest]
        have readyOther (name : String) (unrelated : name ∉ ["j", "diff", "best"]) : ready name = env name := by
          have nj : name ≠ "j" := by intro equal; subst name; simp at unrelated
          have nd : name ≠ "diff" := by intro equal; subst name; simp at unrelated
          have nb : name ≠ "best" := by intro equal; subst name; simp at unrelated
          cases keep : accepted k (count+1) <;> simp [ready, keep, diffed, bound, setLocal, nj, nd, nb]
        obtain ⟨finalEnv, executed, finalBest, untouched⟩ := ih newBest (by omega) ready
          (by rw [readyOther "pent" (by decide)]; exact getTable)
          (by rw [readyOther "pk" (by decide)]; exact getPK) readyBest
        refine ⟨finalEnv, ?_, ?_, ?_⟩
        · apply BlockContinues.range_step (by simp [Calls.rangeContinues])
          simp only [Calls.loopContinuation, show Calls.loopControl innerBody = true from rfl, if_true, next]
          apply assigned.append
          apply BlockContinues.branch_next (flag := false) tested (by simp [Mutable.truth, scalarTruth, cut])
          exact updated.append (BlockContinues.loop_resume executed)
        · simpa only [row_succ, min_eq_right better, newBest] using finalBest
        · intro name unrelated
          rw [untouched name unrelated, readyOther name unrelated]

private def outerBody : List Stmt :=
  [.assign [.name "pk"] (.subscript (.name "pent") (.name "k")),
   .forIn (.name "j") (.call (.name "range")
     [.binOp (.name "k") .sub (.constant (.int 1)), .constant (.int 0), .unaryOp .uSub (.constant (.int 1))]) innerBody []]
def sourceOuterBody : List Stmt :=
  match (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3 with
  | .forIn _ _ body _ :: _ => body
  | _ => []

theorem outerBody_source : outerBody = sourceOuterBody := rfl

/-- The outer source loop scans each larger index once and folds the verified bounded row minimum. -/
theorem outer_result (limit first count best : Nat) (equation : first+count=limit+1) (positive : 0 < first)
    (heap : Mutable.Heap) (address : Nat) (found : heap[address]? = some (.list (Table.entries (limit+1) 0)))
    (env : HeapCalls.Env) (getTable : env "pent" = some (.ref address)) (getBest : env "best" = some (natural best)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "k" first (limit+1) 1 outerBody []] heap finalEnv heap ∧
      finalEnv "best" = some (natural ((List.range' first count).foldl (fun previous k => row k (k-1) previous) best)) ∧
      (∀ name, name ∉ ["k", "pk", "j", "diff", "best"] → finalEnv name = env name) := by
  induction count generalizing first best env with
  | zero =>
      refine ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil, getBest, ?_⟩
      intros; rfl
  | succ count ih =>
      let bound := setLocal env "k" (natural first)
      let ready := setLocal bound "pk" (natural (pentagonal first))
      have assigned : BlockContinues program bound [.assign [.name "pk"] (.subscript (.name "pent") (.name "k"))] heap ready heap :=
        BlockContinues.assign (Evaluates.subscript (by intros; simp)
          (Evaluates.lookup (by simpa [bound, setLocal] using getTable)) (Evaluates.lookup (by simp [bound, setLocal]))
          (Table.lookup heap address limit first (by omega) found))
      obtain ⟨middleEnv, executedRow, middleBest, same⟩ := inner_result limit first (first-1) best (by omega) (by omega)
        heap address found ready (by simpa [ready, bound, setLocal] using getTable) (by simp [ready, setLocal])
          (by simpa [ready, bound, setLocal] using getBest)
      have inner : BlockContinues program ready
          [.forIn (.name "j") (.call (.name "range")
            [.binOp (.name "k") .sub (.constant (.int 1)), .constant (.int 0), .unaryOp .uSub (.constant (.int 1))]) innerBody []]
          heap middleEnv heap := by
        exact BlockContinues.for_range (values := [natural (first-1), natural 0, number (-1)])
          (.cons (Evaluates.binary (av := natural first) (bv := natural 1) (middle := heap) (afterRight := heap)
            (Evaluates.lookup (by simp [ready, bound, setLocal])) Evaluates.constant
            (by simp [Mutable.binary, scalarBinary, Int.natCast_sub (show 1 ≤ first by omega)]))
              (.cons Evaluates.constant (.cons (Evaluates.unary Evaluates.constant rfl) .nil))) rfl executedRow
      obtain ⟨finalEnv, executed, finalBest, untouched⟩ := ih (first+1) (row first (first-1) best)
        (by omega) (by omega) middleEnv
          (by rw [same "pent" (by decide)]; simpa [ready, bound, setLocal] using getTable) middleBest
      refine ⟨finalEnv, ?_, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, show Calls.loopControl outerBody = false from rfl,
          Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using (assigned.append inner).append executed
      · simpa only [List.range'_succ, List.foldl_cons] using finalBest
      · intro name unrelated
        have nk : name ≠ "k" := by intro equal; subst name; simp at unrelated
        have np : name ≠ "pk" := by intro equal; subst name; simp at unrelated
        have others : name ∉ ["j", "diff", "best"] := fun member =>
          unrelated (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ member))
        rw [untouched name unrelated, same name others]
        simp [ready, bound, setLocal, nk, np]

def sourceSetup : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 2).take 1
def sourceLoop : List Stmt :=
  ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3).take 1

/-- The initialized capped search covers zero and positive limits, including empty inner ranges. -/
theorem result (limit : Nat) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (Table.entries (limit+1) 0)))
    (env : HeapCalls.Env) (getTable : env "pent" = some (.ref address)) (getLimit : env "limit" = some (natural limit)) :
    ∃ finalEnv, BlockContinues program env (sourceSetup ++ sourceLoop) heap finalEnv heap ∧
      finalEnv "best" = some (natural (search limit)) := by
  let ready := setLocal env "best" (natural cap)
  have assigned : BlockContinues program env sourceSetup heap ready heap :=
    BlockContinues.assign (Evaluates.binary Evaluates.constant Evaluates.constant rfl)
  have arguments : Evaluates.Arguments program ready [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))]
      heap [natural 2, natural (limit+1)] heap :=
    .cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup (by simpa [ready, setLocal] using getLimit)) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary])) .nil)
  cases limit with
  | zero =>
      refine ⟨ready, assigned.append ?_, by simp [ready, setLocal, search, values]⟩
      exact BlockContinues.for_range arguments rfl (BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil)
  | succ limit =>
      obtain ⟨finalEnv, executed, finalBest, _⟩ := outer_result (limit+1) 2 limit cap (by omega) (by omega)
        heap address found ready (by simpa [ready, setLocal] using getTable) (by simp [ready, setLocal])
      refine ⟨finalEnv, assigned.append ?_, ?_⟩
      · apply BlockContinues.for_range arguments rfl
        simpa only [Int.natCast_add, Int.natCast_one] using executed
      · simpa only [skip_first, Nat.add_sub_cancel] using finalBest

end Search

namespace solve
/-- All source phases preserve the input heap and return the reference's bounded mathematical minimum. -/
theorem result (limit : Nat) (heap : Mutable.Heap) :
    ReturnsValue program "solve" [natural limit] heap (natural (search limit))
      (Table.heap (heap ++ [.list [natural 0]]) (limit+1) 0) := by
  let initial := functionEnv ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 [natural limit] []
  obtain ⟨tableEnv, built, getTable, untouched⟩ := Table.result limit initial heap rfl
  have found := Table.found (heap ++ [.list [natural 0]]) (limit+1) 0
  have addressEqual : (heap ++ [Mutable.Object.list [natural 0]]).length = heap.length+1 := by simp
  rw [addressEqual] at found
  obtain ⟨finalEnv, searched, best⟩ := Search.result limit (Table.heap (heap ++ [.list [natural 0]]) (limit+1) 0)
    (heap.length+1) found tableEnv getTable (by rw [untouched "limit" (by decide)]; rfl)
  apply BlockReturns.function rfl rfl
  apply (built.append searched).returns
  exact BlockReturns.return_value (Evaluates.lookup best)
end solve

private abbrev inputEnv (limit : Int) : ScalarEnv :=
  fun name => if name = "limit" then some (.int limit) else none

private theorem accepted_limit (limit : Int)
    (requires : Calls.Condition program (inputEnv limit) Solvers.P44Ast.requires_solve_0) : 0 ≤ limit := by
  simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare requires rfl

theorem solve.«Equiv(naive)» (limit : Int)
    (requires : Calls.Condition program (inputEnv limit) Solvers.P44Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P44Ast.module "naive" [.int limit]) := by
  have nonnegative := accepted_limit limit requires
  lift limit to Nat using nonnegative
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := [.int limit]) (solve.result limit [])
  obtain ⟨_, returned⟩ := data.«project-euler-statements».data.py.project_euler_statements.p44.naive.result (limit : Int) []
  have reference := HeapCalls.ReturnsValue.scalar_finishes (values := [.int limit]) returned
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem solve.«Supported» :
    (∀ limit : Int, Calls.Condition program (inputEnv limit) Solvers.P44Ast.requires_solve_0 →
      (HeapCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, Calls.Condition program (inputEnv limit) Solvers.P44Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P44Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit requires
    have nonnegative := accepted_limit limit requires
    lift limit to Nat using nonnegative
    exact (HeapCalls.ReturnsValue.scalar_returns (values := [.int limit]) (solve.result limit [])).supported
  · intro limit _
    exact data.«project-euler-statements».data.py.project_euler_statements.p44.naive.«Supported» limit

end solvers.«44»
