import solvers.«61.world.source»
import ProjectEulerSolutions.Lib.HeapLoop
import ProjectEulerSolutions.Lib.HeapSequenceFold
import lang.py.FormalSW.MembershipCalls

set_option Elab.async false

namespace solvers.«61».dfs
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

def unusedFilter : Expr := .compare (.name "other") [(.notEq, .name "t")]
def unusedExpression : Expr := .listComp (.name "other") [("other", .name "remaining", [unusedFilter])]
def extendedPath : Expr := .binOp (.name "path") .add (.list [.name "nxt"])
def childArguments : List Expr := [extendedPath, .name "unused", .name "__formalsw_cell_solve_pref_map"]
def childCall : Expr := .call (.name "__formalsw_local_solve_dfs") childArguments
def innerBody : List Stmt :=
  [.ifThen (.membership (.name "nxt") (.name "path") false) [.continueLoop] [],
   .assign [.name "best"] (.builtinCall .max [.name "best", childCall])]
def outerBody : List Stmt :=
  [.assign [.name "unused"] unusedExpression, .forIn (.name "nxt") nextValues innerBody []]
def suffixExpression : Expr := .binOp
  (.subscript (.name "path") (.unaryOp .uSub (.constant (.int 1)))) .mod (.constant (.int 100))
def code : List Stmt :=
  [.ifThen exhausted baseBody [], .assign [.name "best"] (.constant (.int 0)),
   .assign [.name "suffix"] suffixExpression,
   .forIn (.name "t") (.name "remaining") outerBody [], .return (.name "best")]

theorem code_source :
    executableBody ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2 = code := rfl

def kinds (values : List Int) : List Mutable.Value := values.map (fun value : Int => number value)
private def unusedBody (collector : Nat) : List Stmt :=
  [.ifThen unusedFilter [.comprehensionAppend collector (.name "other")] []]

private theorem unused_loop (all remaining : List Int) (position : Nat) (suffix : all.drop position = remaining)
    (kind : Int) (base : Mutable.Heap) (address : Nat) (found : base[address]? = some (.list (kinds all)))
    (before : List Int) (env : HeapCalls.Env) (getKind : env "t" = some (number kind)) (globals : Mutable.Globals) :
    ∃ finalEnv, BlockContinues program env [.sequenceStepLoop "other" (.ref address) position (unusedBody base.length) []]
      (HeapCollection.working base (kinds before) []) finalEnv
      (HeapCollection.working base (kinds (before ++ remaining.filter (fun other => decide (other ≠ kind)))) []) globals := by
  have sequence (collected : List Int) :
      Mutable.sequence (sequenceValue (.ref address)) (HeapCollection.working base (kinds collected) []) =
        some (.ok (kinds all), HeapCollection.working base (kinds collected) []) := by
    have kept := HeapCollection.prefix_found base [.list (kinds collected)] address _ found
    simpa only [HeapCollection.working, List.append_nil] using
      (show Mutable.sequence (.ref address) (base ++ [.list (kinds collected)]) =
        some (.ok (kinds all), base ++ [.list (kinds collected)]) by simp [Mutable.sequence, kept])
  induction remaining generalizing position before env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.filter_nil, List.append_nil]
      apply BlockContinues.sequence_done (sequence before)
      · have missing : all[position]? = none := by rw [← List.head?_drop, suffix]; rfl
        simp [kinds, List.getElem?_map, missing]
      · exact BlockContinues.nil
  | cons other rest ih =>
      have selected : all[position]? = some other := by rw [← List.head?_drop, suffix]; rfl
      let bound := setLocal env "other" (number other)
      have boundKind : bound "t" = some (number kind) := by simpa [bound, setLocal] using getKind
      have tested : Evaluates program bound unusedFilter (HeapCollection.working base (kinds before) [])
          (flag (decide (other ≠ kind))) (HeapCollection.working base (kinds before) []) globals := by
        apply Evaluates.compare (middle := HeapCollection.working base (kinds before) [])
          (afterRight := HeapCollection.working base (kinds before) [])
          (finalHeap := HeapCollection.working base (kinds before) [])
          (Evaluates.lookup (by rfl)) (Evaluates.lookup boundKind)
        simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]
      have nextSuffix : all.drop (position + 1) = rest := by
        simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]
      by_cases different : other ≠ kind
      · rw [show decide (other ≠ kind) = true from decide_eq_true different] at tested
        have emitted : BlockContinues program bound (unusedBody base.length)
            (HeapCollection.working base (kinds before) []) bound
            (HeapCollection.working base (kinds (before ++ [other])) []) globals := by
          apply BlockContinues.branch (flag := true) tested rfl
          have stored := HeapCollection.emit (program := program) base (kinds before) [] [] (number other)
            (by simpa only [List.append_nil] using
              (Evaluates.lookup (module := program) (globals := globals)
                (heap := HeapCollection.working base (kinds before) []) (show bound "other" = some (number other) from rfl)))
          simpa only [kinds, List.map_append, List.map_cons, List.map_nil, List.append_nil] using stored
        obtain ⟨finalEnv, continued⟩ := ih (position + 1) nextSuffix (before ++ [other]) bound boundKind
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.sequence_step (value := number other) (sequence before) (by simp [kinds, List.getElem?_map, selected])
        simpa [unusedBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          List.filter_cons, different, List.append_assoc] using emitted.append continued
      · rw [show decide (other ≠ kind) = false from decide_eq_false different] at tested
        have skipped : BlockContinues program bound (unusedBody base.length)
            (HeapCollection.working base (kinds before) []) bound (HeapCollection.working base (kinds before) []) globals :=
          BlockContinues.branch (flag := false) tested rfl .nil
        obtain ⟨finalEnv, continued⟩ := ih (position + 1) nextSuffix before bound boundKind
        refine ⟨finalEnv, ?_⟩
        apply BlockContinues.sequence_step (value := number other) (sequence before) (by simp [kinds, List.getElem?_map, selected])
        simpa [unusedBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          List.filter_cons, different] using skipped.append continued

/-- Filtering the remaining types creates one fresh list and preserves every old object. -/
theorem unused_evaluate (all : List Int) (kind : Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (kinds all))) (env : HeapCalls.Env)
    (getRemaining : env "remaining" = some (.ref address)) (getKind : env "t" = some (number kind))
    (globals : Mutable.Globals := []) :
    Evaluates program env unusedExpression heap (.ref heap.length)
      (heap ++ [.list (kinds (all.filter (fun other => decide (other ≠ kind))))]) globals := by
  let masked := comprehensionEnv env [("other", .name "remaining", [unusedFilter])]
  have maskedKind : masked "t" = some (number kind) := by
    simpa [masked, comprehensionEnv, setLocal] using getKind
  obtain ⟨finalEnv, executed⟩ := unused_loop all all 0 rfl kind heap address found [] masked maskedKind globals
  apply Evaluates.list_comp_sequence (items := kinds all) (finalEnv := finalEnv)
    (by intro expressions impossible; cases impossible) (Evaluates.lookup getRemaining)
    (by simp [Mutable.sequence, found])
  simpa only [masked, comprehensionFilters, comprehensionBody, unusedBody, HeapCollection.working,
    kinds, List.map_nil, List.nil_append, List.append_nil] using executed

def childHeap (heap : Mutable.Heap) (values : List Int) (value : Int) : Mutable.Heap :=
  heap ++ [.list [number value], .list (numbers (values ++ [value]))]

theorem path_argument_evaluate (values : List Int) (heap : Mutable.Heap) (path : Nat)
    (found : heap[path]? = some (.list (numbers values))) (value : Int) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (getValue : env "nxt" = some (number value))
    (globals : Mutable.Globals := []) :
    Evaluates program env extendedPath heap (.ref (heap.length + 1)) (childHeap heap values value) globals := by
  have singleton : Evaluates program env (.list [.name "nxt"]) heap (.ref heap.length)
      (heap ++ [.list [number value]]) globals := Evaluates.list (.cons (Evaluates.lookup getValue) .nil)
  have kept := HeapCollection.prefix_found heap [.list [number value]] path _ found
  apply Evaluates.binary (middle := heap) (afterRight := heap ++ [.list [number value]])
    (Evaluates.lookup getPath) singleton
  simp [Mutable.binary, show (BinOp.add == BinOp.add) = true from rfl, Mutable.array, kept,
    Mutable.allocate, Except.map, childHeap, numbers, List.append_assoc]

theorem child_arguments_evaluate (values : List Int) (heap : Mutable.Heap) (path unused prefixCell : Nat)
    (found : heap[path]? = some (.list (numbers values))) (value : Int) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (getValue : env "nxt" = some (number value))
    (getUnused : env "unused" = some (.ref unused))
    (getCell : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell)) (globals : Mutable.Globals := []) :
    Evaluates.Arguments program env childArguments heap [.ref (heap.length + 1), .ref unused, .ref prefixCell]
      (childHeap heap values value) globals :=
  .cons (path_argument_evaluate values heap path found value env getPath getValue globals)
    (.cons (Evaluates.lookup getUnused) (.cons (Evaluates.lookup getCell) .nil))

/-- Source parameters and captured tables describe the same logical reference search. -/
structure SearchState (heap : Mutable.Heap) (path remainingAddress prefixCell : Nat) (maxSides : Int)
    (values : List Int) (remaining : List Int) where
  world : World heap prefixCell maxSides
  pathFound : heap[path]? = some (.list (numbers values))
  remainingFound : heap[remainingAddress]? = some (.list (kinds remaining))
  validKinds : ∀ kind ∈ remaining, 3 ≤ kind ∧ kind ≤ maxSides

def SearchState.append (state : SearchState heap path remainingAddress prefixCell maxSides values remaining)
    (extra : Mutable.Heap) :
    SearchState (heap ++ extra) path remainingAddress prefixCell maxSides values remaining where
  world := state.world.append extra
  pathFound := HeapCollection.prefix_found _ _ _ _ state.pathFound
  remainingFound := HeapCollection.prefix_found _ _ _ _ state.remainingFound
  validKinds := state.validKinds

def SearchState.child (state : SearchState heap path remainingAddress prefixCell maxSides values remaining)
    (kind : Int) (value : Int) (unused : Nat)
    (unusedFound : heap[unused]? = some (.list (kinds (remaining.filter (fun other => decide (other ≠ kind)))))) :
    SearchState (childHeap heap values value) (heap.length + 1) unused prefixCell maxSides (values ++ [value])
      (remaining.filter (fun other => decide (other ≠ kind))) where
  world := state.world.append [.list [number value], .list (numbers (values ++ [value]))]
  pathFound := by simp [childHeap]
  remainingFound := HeapCollection.prefix_found _ _ _ _ unusedFound
  validKinds := by
    intro other member
    exact state.validKinds other (List.mem_filter.mp member).1

theorem membership_evaluate (values : List Int) (heap : Mutable.Heap) (path : Nat)
    (found : heap[path]? = some (.list (numbers values))) (value : Int) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (getValue : env "nxt" = some (number value))
    (globals : Mutable.Globals := []) :
    Evaluates program env (.membership (.name "nxt") (.name "path") false) heap
      (flag (decide (value ∈ values))) heap globals := by
  apply Evaluates.membership (.cons (Evaluates.lookup getValue) (.cons (Evaluates.lookup getPath) .nil))
  exact HeapCalls.membership_list_ints heap path value values found false

def lastExpression : Expr := .subscript (.name "path") (.unaryOp .uSub (.constant (.int 1)))

theorem last_evaluate (values : List Int) (last : Int) (atEnd : values.getLast? = some last)
    (heap : Mutable.Heap) (path : Nat) (found : heap[path]? = some (.list (numbers values)))
    (env : HeapCalls.Env) (getPath : env "path" = some (.ref path)) (globals : Mutable.Globals := []) :
    Evaluates program env lastExpression heap (number last) heap globals := by
  have nonempty : values ≠ [] := by intro empty; simp [empty] at atEnd
  have positive : 0 < values.length := List.length_pos_iff.mpr nonempty
  have tail : values[values.length - 1]? = some last := by rw [← List.getLast?_eq_getElem?]; exact atEnd
  apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getPath)
    (Evaluates.unary (middle := heap) (finalHeap := heap) (value := number 1) Evaluates.constant rfl)
  have offset : ((-1 : Int) + values.length).toNat = values.length - 1 := by omega
  have indexed : Mutable.index (numbers values) (number (-1)) heap = some (.ok (values.length - 1), heap) := by
    simp [Mutable.index, numbers, nonempty, offset]
  have mapped : (numbers values)[values.length - 1]? = some (number last) := by simp [numbers, tail]
  simp [Mutable.getItem, found, indexed, Except.map, mapped]

theorem suffix_evaluate (values : List Int) (last : Int) (atEnd : values.getLast? = some last)
    (heap : Mutable.Heap) (path : Nat) (found : heap[path]? = some (.list (numbers values)))
    (env : HeapCalls.Env) (getPath : env "path" = some (.ref path)) (globals : Mutable.Globals := []) :
    Evaluates program env suffixExpression heap (number (last % 100)) heap globals := by
  apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
    (last_evaluate values last atEnd heap path found env getPath globals) Evaluates.constant
  simp [Mutable.binary, scalarBinary, number]

theorem maximum_evaluate (values : List Int) (heap : Mutable.Heap) (path unused prefixCell : Nat)
    (found : heap[path]? = some (.list (numbers values))) (value best child : Int) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (getValue : env "nxt" = some (number value))
    (getUnused : env "unused" = some (.ref unused)) (getBest : env "best" = some (number best))
    (getCell : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell))
    (returned : ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref (heap.length + 1), .ref unused, .ref prefixCell] (childHeap heap values value)
      (number child) finalHeap globals) :
    Evaluates program env (.builtinCall .max [.name "best", childCall]) heap (number (max best child)) finalHeap globals := by
  apply Evaluates.builtin (.cons (Evaluates.lookup getBest)
    (.cons (Evaluates.call
      (child_arguments_evaluate values heap path unused prefixCell found value env getPath getValue getUnused getCell globals)
      returned) .nil))
  by_cases better : best < child
  · simp [primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare, evalCmpOp, scalarResult,
      better, Int.max_eq_right (by omega : best ≤ child)]
  · simp [primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare, evalCmpOp, scalarResult,
      better, Int.max_eq_left (by omega : child ≤ best)]

theorem inner_used (values : List Int) (heap : Mutable.Heap) (path : Nat)
    (found : heap[path]? = some (.list (numbers values))) (value : Int) (used : value ∈ values)
    (env : HeapCalls.Env) (getPath : env "path" = some (.ref path)) (getValue : env "nxt" = some (number value))
    (globals : Mutable.Globals := []) : HeapLoop.Continues program env innerBody heap env heap globals := by
  have seen := membership_evaluate values heap path found value env getPath getValue globals
  rw [show decide (value ∈ values) = true from decide_eq_true used] at seen
  apply HeapLoop.branch (flag := true) seen rfl
  apply HeapLoop.skip
  intros
  rfl

theorem inner_updated (values : List Int)
    (heap : Mutable.Heap) (path unused prefixCell : Nat) (found : heap[path]? = some (.list (numbers values)))
    (value best child : Int) (fresh : value ∉ values)
    (env : HeapCalls.Env) (getPath : env "path" = some (.ref path)) (getValue : env "nxt" = some (number value))
    (getUnused : env "unused" = some (.ref unused)) (getBest : env "best" = some (number best))
    (getCell : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell))
    (returned : ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref (heap.length + 1), .ref unused, .ref prefixCell] (childHeap heap values value)
      (number child) finalHeap globals) :
    HeapLoop.Continues program env innerBody heap (setLocal env "best" (number (max best child))) finalHeap globals := by
  have seen := membership_evaluate values heap path found value env getPath getValue globals
  rw [show decide (value ∈ values) = false from decide_eq_false fresh] at seen
  apply HeapLoop.branch (flag := false) seen rfl
  have assigned := BlockContinues.assign (name := "best")
    (maximum_evaluate values heap path unused prefixCell found value best child env getPath getValue getUnused getBest getCell returned)
  have resumed := HeapLoop.prepend assigned (HeapLoop.nil program _ _ globals)
  simpa only [List.append_nil] using resumed

structure Bindings (env : HeapCalls.Env) (path remainingAddress prefixCell : Nat) : Prop where
  getPath : env "path" = some (.ref path)
  getRemaining : env "remaining" = some (.ref remainingAddress)
  getCell : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell)

theorem Bindings.local (bound : Bindings env path remainingAddress prefixCell) (name : String) (value : Mutable.Value)
    (fresh : name ∉ ["path", "remaining", "__formalsw_cell_solve_pref_map"]) :
    Bindings (setLocal env name value) path remainingAddress prefixCell := by
  simp only [List.mem_cons, not_or] at fresh
  refine ⟨?_, ?_, ?_⟩ <;> simp_all [setLocal, eq_comm, bound.getPath, bound.getRemaining, bound.getCell]

theorem Bindings.function (path remainingAddress prefixCell : Nat) (globals : Mutable.Globals) :
    Bindings (functionEnv
      ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
      ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
      [.ref path, .ref remainingAddress, .ref prefixCell] globals) path remainingAddress prefixCell := by
  constructor <;> cases globals <;> rfl

end solvers.«61».dfs
