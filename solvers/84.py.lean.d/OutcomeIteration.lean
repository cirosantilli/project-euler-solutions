import solvers.«84.py.lean.d».OutcomeArithmetic

namespace solvers.«84».outcomes
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def ordinaryCode : Stmt := .ifThen (.compare (.name "count") [(.gt, .name "double")])
  [.expr (.call (.attribute (.name "outcomes") "append")
    [.tuple [.name "move", .constant (.bool false), .binOp (.name "count") .sub (.name "double")]])] []
def doubleCode : Stmt := .ifThen (.name "double")
  [.expr (.call (.attribute (.name "outcomes") "append") [.tuple [.name "move", .constant (.bool true), .constant (.int 1)]])] []
def code : List Stmt := [.assign [.name "count"] countExpression, .assign [.name "double"] doubleExpression, ordinaryCode, doubleCode]

theorem source : (executableBody ((findFunctionIn program.body "build_transition").getD ([], [])).2)[(2 : Nat)]?.bind
    (fun statement => match statement with | .forIn _ _ body _ => some body | _ => none) = some code := rfl

def ordinary (sides move : Int) : List Outcome :=
  if double move < count sides move then [⟨move, false, count sides move - double move⟩] else []
def doubled (move : Int) : List Outcome := if move % 2 = 0 then [⟨move, true, 1⟩] else []
def perMove (sides move : Int) : List Outcome := ordinary sides move ++ doubled move
def locals (env : HeapCalls.Env) (sides move : Int) : HeapCalls.Env :=
  setLocal (setLocal env "count" (number (count sides move))) "double" (number (double move))

theorem ordinary_emit (sides move : Int) (base : Mutable.Heap) (values : List Outcome) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getOutcomes : env "outcomes" = some (.ref base.length)) (getMove : env "move" = some (number move))
    (getCount : env "count" = some (number (count sides move))) (getDouble : env "double" = some (number (double move))) :
    BlockContinues program env [ordinaryCode] (heap base values) env (heap base (values ++ ordinary sides move)) globals := by
  have tested : Evaluates program env (.compare (.name "count") [(.gt, .name "double")]) (heap base values)
      (.scalar (.bool (decide (double move < count sides move)))) (heap base values) globals :=
    Evaluates.compare (Evaluates.lookup getCount) (Evaluates.lookup getDouble) (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  apply BlockContinues.branch tested rfl
  by_cases accepted : double move < count sides move
  · simp only [accepted, decide_true, ite_true, ordinary]
    apply emit base values ⟨move, false, count sides move - double move⟩ env _ globals getOutcomes
    exact .cons (Evaluates.lookup getMove) (.cons Evaluates.constant (.cons
      (sub env (heap base values) globals _ _ _ _ (Evaluates.lookup getCount) (Evaluates.lookup getDouble)) .nil))
  · simpa [ordinary, accepted] using (BlockContinues.nil (module := program) (env := env) (heap := heap base values) (globals := globals))

theorem double_emit (move : Int) (base : Mutable.Heap) (values : List Outcome) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getOutcomes : env "outcomes" = some (.ref base.length)) (getMove : env "move" = some (number move))
    (getDouble : env "double" = some (number (double move))) :
    BlockContinues program env [doubleCode] (heap base values) env (heap base (values ++ doubled move)) globals := by
  apply BlockContinues.branch (flag := decide (move % 2 = 0)) (Evaluates.lookup getDouble)
    (by by_cases even : move % 2 = 0 <;> simp [Mutable.truth, scalarTruth, double, even])
  by_cases even : move % 2 = 0
  · simp only [even, decide_true, ite_true, doubled]
    apply emit base values ⟨move, true, 1⟩ env _ globals getOutcomes
    exact .cons (Evaluates.lookup getMove) (.cons Evaluates.constant (.cons Evaluates.constant .nil))
  · simpa [doubled, even] using (BlockContinues.nil (module := program) (env := env) (heap := heap base values) (globals := globals))

theorem iteration (sides move : Int) (base : Mutable.Heap) (values : List Outcome) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getSides : env "dice_sides" = some (number sides)) (getMove : env "move" = some (number move))
    (getOutcomes : env "outcomes" = some (.ref base.length)) :
    BlockContinues program env code (heap base values) (locals env sides move) (heap base (values ++ perMove sides move)) globals := by
  have counted := BlockContinues.assign (name := "count") (count_result sides move env (heap base values) globals getSides getMove)
  have marked := BlockContinues.assign (name := "double") (double_result move (setLocal env "count" (number (count sides move)))
    (heap base values) globals (by simpa [setLocal] using getMove))
  have keptOutcomes : locals env sides move "outcomes" = some (.ref base.length) := by simpa [locals, setLocal] using getOutcomes
  have keptMove : locals env sides move "move" = some (number move) := by simpa [locals, setLocal] using getMove
  have ordinaryRun := ordinary_emit sides move base values (locals env sides move) globals keptOutcomes keptMove
    (by simp [locals, setLocal]) (by simp [locals, setLocal])
  have doubles := double_emit move base (values ++ ordinary sides move) (locals env sides move) globals keptOutcomes keptMove
    (by simp [locals, setLocal])
  simpa only [code, perMove, List.append_assoc, List.cons_append, List.nil_append] using counted.append (marked.append (ordinaryRun.append doubles))

end solvers.«84».outcomes
