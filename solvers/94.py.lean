import «.formalsw».solvers.«94.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p94.py»
import ProjectEulerSolutions.Lib.AlmostEquilateral
import ProjectEulerSolutions.Lib.HeapCollection
import lang.py.FormalSW.SumCalls

namespace solvers.«94»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.AlmostEquilateral
open ProjectEulerSolutions.HeapCollection
set_option Elab.async false

abbrev program := Solvers.P94Ast.module
abbrev number (n : Int) : Mutable.Value := .scalar (.int n)

namespace generate_almost_equilateral_perimeters

abbrev longerCode : List Stmt :=
  [.assign [.name "a"] (.binOp (.binOp (.name "x") .add (.constant (.int 1))) .floorDiv (.constant (.int 3))),
   .assign [.name "p"] (.binOp (.binOp (.constant (.int 3)) .mult (.name "a")) .add (.constant (.int 1)))]
abbrev shorterCode : List Stmt :=
  [.assign [.name "a"] (.binOp (.binOp (.name "x") .sub (.constant (.int 1))) .floorDiv (.constant (.int 3))),
   .assign [.name "p"] (.binOp (.binOp (.constant (.int 3)) .mult (.name "a")) .sub (.constant (.int 1)))]
abbrev selectCode : List Stmt :=
  [.ifThen (.compare (.binOp (.name "x") .mod (.constant (.int 3))) [(.eq, .constant (.int 2))])
    longerCode shorterCode]
abbrev advanceCode : List Stmt :=
  [.expr (.call (.attribute (.name "perimeters") "append") [.name "p"]),
   .assign [.tuple [.name "x", .name "y"]] (.tuple
     [.binOp (.binOp (.constant (.int 2)) .mult (.name "x")) .add
         (.binOp (.constant (.int 3)) .mult (.name "y")),
      .binOp (.name "x") .add (.binOp (.constant (.int 2)) .mult (.name "y"))])]
abbrev body : List Stmt := selectCode ++
  [.ifThen (.compare (.name "p") [(.gt, .name "limit")]) [.breakLoop] []] ++ advanceCode
abbrev loop : Stmt := .whileDo (.constant (.bool true)) body []

theorem source_body : executableBody ((findFunctionIn program.body "generate_almost_equilateral_perimeters").getD ([], [])).2 =
    [.assign [.tuple [.name "x", .name "y"]] (.tuple [.constant (.int 14), .constant (.int 8)]),
     .annAssign (.name "perimeters") (.subscript (.name "List") (.name "int")) (some (.list [])),
     loop, .return (.name "perimeters")] := rfl

theorem select_result (x : Int) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getX : env "x" = some (number x)) :
    ∃ nextEnv, BlockContinues program env selectCode heap nextEnv heap globals ∧
      nextEnv "p" = some (number (sourcePerimeter x)) ∧
      ∀ name, name ≠ "a" → name ≠ "p" → nextEnv name = env name := by
  have test : Evaluates program env
      (.compare (.binOp (.name "x") .mod (.constant (.int 3))) [(.eq, .constant (.int 2))])
      heap (.scalar (.bool (decide (x % 3 = 2)))) heap globals := by
    apply Evaluates.compare (av := number (x % 3)) (bv := number 2)
    · exact Evaluates.binary (Evaluates.lookup getX) Evaluates.constant rfl
    · exact Evaluates.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases longer : x % 3 = 2
  · let withA := setLocal env "a" (number ((x + 1) / 3))
    let ready := setLocal withA "p" (number (3 * ((x + 1) / 3) + 1))
    refine ⟨ready, ?_, by simp [ready, setLocal, sourcePerimeter, longer], ?_⟩
    · apply BlockContinues.branch (flag := true) (by simpa [longer] using test) rfl
      apply BlockContinues.append (BlockContinues.assign
        (Evaluates.binary (Evaluates.binary (Evaluates.lookup getX) Evaluates.constant rfl) Evaluates.constant rfl))
      exact BlockContinues.assign (Evaluates.binary
        (Evaluates.binary Evaluates.constant (Evaluates.lookup rfl) rfl) Evaluates.constant rfl)
    · intro name notA notP
      simp [ready, withA, setLocal, notA, notP]
  · let withA := setLocal env "a" (number ((x - 1) / 3))
    let ready := setLocal withA "p" (number (3 * ((x - 1) / 3) - 1))
    refine ⟨ready, ?_, by simp [ready, setLocal, sourcePerimeter, longer], ?_⟩
    · apply BlockContinues.branch (flag := false) (by simpa [longer] using test) rfl
      apply BlockContinues.append (BlockContinues.assign
        (Evaluates.binary (Evaluates.binary (Evaluates.lookup getX) Evaluates.constant rfl) Evaluates.constant rfl))
      exact BlockContinues.assign (Evaluates.binary
        (Evaluates.binary Evaluates.constant (Evaluates.lookup rfl) rfl) Evaluates.constant rfl)
    · intro name notA notP
      simp [ready, withA, setLocal, notA, notP]

/-- Execute every actual append and tuple update until the increasing perimeter exceeds the bound. -/
theorem loop_result (limit : Int) (n : Nat) (nontrivial : 2 ≤ n)
    (base garbage : Mutable.Heap) (before : List Int) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getX : env "x" = some (number (2 * (solution n).x)))
    (getY : env "y" = some (number (2 * (solution n).y)))
    (getLimit : env "limit" = some (number limit))
    (getList : env "perimeters" = some (.ref base.length)) :
    ∃ finalGarbage, BlockReturns program env [loop, .return (.name "perimeters")]
      (working base (before.map number) garbage) (.ref base.length)
      (working base ((before ++ perimetersFrom limit n nontrivial).map number) finalGarbage) globals := by
  let heap := working base (before.map number) garbage
  obtain ⟨ready, selected, getP, kept⟩ := select_result (2 * (solution n).x) heap env globals getX
  rw [sourcePerimeter_eq] at getP
  have readyX := (kept "x" (by decide) (by decide)).trans getX
  have readyY := (kept "y" (by decide) (by decide)).trans getY
  have readyLimit := (kept "limit" (by decide) (by decide)).trans getLimit
  have readyList := (kept "perimeters" (by decide) (by decide)).trans getList
  have test : Evaluates program ready (.compare (.name "p") [(.gt, .name "limit")]) heap
      (.scalar (.bool (decide (limit < perimeter n)))) heap globals :=
    Evaluates.compare (Evaluates.lookup getP) (Evaluates.lookup readyLimit)
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases done : limit < perimeter n
  · refine ⟨garbage, ?_⟩
    rw [perimetersFrom, if_pos done, List.append_nil]
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply selected.returns
    apply BlockReturns.branch (flag := true) (by simpa [done] using test) rfl
    apply BlockReturns.break_loop (next := loop) (following := [.return (.name "perimeters")]) rfl
    exact BlockReturns.return_value (Evaluates.lookup readyList)
  · let x := 2 * (solution n).x
    let y := 2 * (solution n).y
    let nextX := 2 * x + 3 * y
    let nextY := x + 2 * y
    let afterAppend := working base ((before ++ [perimeter n]).map number) garbage
    let nextEnv := assignNames ready ["x", "y"] [number nextX, number nextY]
    let nextGarbage := garbage ++ [.tuple [number nextX, number nextY]]
    have stepX : nextX = 2 * (solution (n + 1)).x := by
      dsimp [nextX, x, y]
      rw [solution_next_x]
      ring
    have stepY : nextY = 2 * (solution (n + 1)).y := by
      dsimp [nextY, x, y]
      rw [solution_next_y]
      ring
    obtain ⟨finalGarbage, rest⟩ := loop_result limit (n + 1) (by omega) base nextGarbage
      (before ++ [perimeter n]) nextEnv globals
      (by simp [nextEnv, assignNames, setLocal, stepX])
      (by simp [nextEnv, assignNames, setLocal, stepY])
      (by simpa [nextEnv, assignNames, setLocal] using readyLimit)
      (by simpa [nextEnv, assignNames, setLocal] using readyList)
    have appended : Evaluates program ready (.call (.attribute (.name "perimeters") "append") [.name "p"])
        heap (.scalar .none) afterAppend globals := by
      simpa only [heap, store, afterAppend, List.map_append, List.map_cons, List.map_nil] using
        Evaluates.append (module := program) (globals := globals) (Evaluates.lookup readyList)
          (found base (before.map number) garbage) (Evaluates.lookup getP)
          (found base (before.map number) garbage)
    have pair : Evaluates program ready (.tuple
        [.binOp (.binOp (.constant (.int 2)) .mult (.name "x")) .add
            (.binOp (.constant (.int 3)) .mult (.name "y")),
         .binOp (.name "x") .add (.binOp (.constant (.int 2)) .mult (.name "y"))])
        afterAppend (.ref afterAppend.length) (afterAppend ++ [.tuple [number nextX, number nextY]]) globals := by
      apply Evaluates.tuple (nonempty := by simp)
      refine .cons (middle := afterAppend) ?_ (.cons (middle := afterAppend) ?_ .nil)
      · exact Evaluates.binary (Evaluates.binary Evaluates.constant (Evaluates.lookup readyX) rfl)
          (Evaluates.binary Evaluates.constant (Evaluates.lookup readyY) rfl) rfl
      · exact Evaluates.binary (Evaluates.lookup readyX)
          (Evaluates.binary Evaluates.constant (Evaluates.lookup readyY) rfl) rfl
    refine ⟨finalGarbage, ?_⟩
    rw [perimetersFrom, if_neg done]
    apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
    apply selected.returns
    apply BlockReturns.branch (flag := false) (by simpa [done] using test) rfl
    apply BlockReturns.discard appended
    apply BlockReturns.unpack (items := [number nextX, number nextY])
      (afterUnpack := afterAppend ++ [.tuple [number nextX, number nextY]]) rfl pair
    · simp [Mutable.sequence]
    · rfl
    · change BlockReturns program nextEnv [.loopResume loop, .return (.name "perimeters")]
        (afterAppend ++ [.tuple [number nextX, number nextY]]) (.ref base.length)
        (working base ((before ++ perimeter n :: perimetersFrom limit (n + 1) (by omega)).map number)
          finalGarbage) globals
      apply BlockReturns.loop_resume
      simpa only [afterAppend, nextGarbage, append, List.append_assoc, List.singleton_append] using rest
termination_by (limit + 3 - 2 * (solution n).x).toNat
decreasing_by
  have bounds := perimeter_bounds n
  have growth : (solution n).x < (solution (n + 1)).x := solution_x_strictMono (Nat.lt_succ_self n)
  omega

theorem result (limit : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ garbage, ReturnsValue program "generate_almost_equilateral_perimeters" [number limit] heap
      (.ref (heap.length + 1))
      (working (heap ++ [.tuple [number 14, number 8]])
        ((perimetersFrom limit 2 (by decide)).map number) garbage) globals := by
  let args := ((findFunctionIn program.body "generate_almost_equilateral_perimeters").getD ([], [])).1
  let source := ((findFunctionIn program.body "generate_almost_equilateral_perimeters").getD ([], [])).2
  let entry := functionEnv args source [number limit] globals
  let paired := assignNames entry ["x", "y"] [number 14, number 8]
  let base := heap ++ [.tuple [number 14, number 8]]
  let ready := setLocal paired "perimeters" (.ref base.length)
  obtain ⟨garbage, executed⟩ := loop_result limit 2 (by decide) base [] [] ready globals
    (by rfl)
    (by rfl)
    (by cases globals <;> rfl) (by simp [ready, setLocal])
  refine ⟨garbage, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.unpack (items := [number 14, number 8]) (afterUnpack := base) rfl
  · exact Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (by simp)
  · simp [Mutable.sequence, base]
  · rfl
  apply (BlockContinues.annAssign (name := "perimeters") (Evaluates.list .nil)).returns
  simpa only [List.nil_append, List.map_nil, ready, base, working, List.length_append,
    List.length_singleton] using executed

theorem «Returns» (limit : Int) :
    (HeapCalls.call program "generate_almost_equilateral_perimeters" [number limit] []).Returns := by
  obtain ⟨_, executed⟩ := result limit []
  exact executed.returns

theorem «Supported» (limit : Int) :
    (HeapCalls.call program "generate_almost_equilateral_perimeters" [number limit] []).Supported :=
  («Returns» limit).supported

end generate_almost_equilateral_perimeters

namespace solve

def Verified (limit : Int) : Prop :=
  (HeapCalls.scalarFunction program "solve" [.int limit]).Returns ∧
  (HeapCalls.scalarFunction P94Ast.module "naive" [.int limit]).Returns ∧
  (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
    (HeapCalls.scalarFunction P94Ast.module "naive" [.int limit])

theorem result (limit : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ finalHeap, ReturnsValue program "solve" [number limit] heap
      (number ((perimetersFrom limit 2 (by decide)).foldl Int.add 0)) finalHeap globals := by
  obtain ⟨garbage, generated⟩ := generate_almost_equilateral_perimeters.result limit heap globals
  refine ⟨working (heap ++ [.tuple [number 14, number 8]])
    ((perimetersFrom limit 2 (by decide)).map number) garbage, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.sum_integers
    (.cons (Evaluates.call (.cons (Evaluates.lookup (by cases globals <;> rfl)) .nil) generated)
      (.cons Evaluates.constant .nil))
  simp [Mutable.sequence, working, number]

theorem «Returns» (limit : Int) : (HeapCalls.scalarFunction program "solve" [.int limit]).Returns := by
  obtain ⟨_, executed⟩ := result limit []
  exact executed.scalar_returns (values := [.int limit])

theorem «Equiv(naive)» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Equiv
      (HeapCalls.scalarFunction P94Ast.module "naive" [.int limit]) := by
  obtain ⟨_, left⟩ := result limit []
  obtain ⟨_, right⟩ := data.«project-euler-statements».data.py.project_euler_statements.p94.naive.result limit [] []
  have same : (reference limit).sum = (perimetersFrom limit 2 (by decide)).foldl Int.add 0 := by
    rw [reference_sum, List.sum_eq_foldl]
    rfl
  rw [same] at right
  have optimized := left.scalar_finishes (values := [.int limit])
  have reference := right.scalar_finishes (values := [.int limit])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» (limit : Int) :
    (HeapCalls.scalarFunction program "solve" [.int limit]).Supported :=
  («Returns» limit).supported

end solve

theorem solve_source_equivalence (limit : Int) : solve.Verified limit := by
  obtain ⟨_, reference⟩ := data.«project-euler-statements».data.py.project_euler_statements.p94.naive.result limit [] []
  exact ⟨solve.«Returns» limit, reference.scalar_returns (values := [.int limit]), solve.«Equiv(naive)» limit⟩

end solvers.«94»
