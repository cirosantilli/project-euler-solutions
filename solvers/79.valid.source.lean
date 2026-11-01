import solvers.«79.attempt.source»
import lang.py.FormalSW.DigitCalls

/-! Valid Unicode attempts pass the source guard and unpack before updating the graph. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def invalidTest : Expr := .boolOp .or
  [.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 3))],
   .unaryOp .not (.call (.attribute (.name "s") "isdigit") [])]

def invalidBody : List Stmt := match (attemptBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

def unpackBody : List Stmt :=
  [.assign [.tuple [.name "a", .name "b", .name "c"]]
    (.tuple [.subscript (.name "s") (.constant (.int 0)),
      .subscript (.name "s") (.constant (.int 1)),
      .subscript (.name "s") (.constant (.int 2))])]

theorem attempt_parts : attemptBody =
    [.ifThen invalidTest invalidBody []] ++ unpackBody ++ updatesBody := rfl

theorem valid_guard (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) (length : s.length = 3)
    (digits : Strings.isDigit s = true) :
    BlockContinues program env [.ifThen invalidTest invalidBody []] heap env heap := by
  have sized : Evaluates program env (.builtinCall .len [.name "s"]) heap (number 3) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getS) .nil)
    simp [primitive, Mutable.length, length, number, text]
  have compared : Evaluates program env
      (.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 3))])
      heap (.scalar (.bool false)) heap := by
    exact Evaluates.compare sized Evaluates.constant rfl
  have tested : Evaluates program env (.call (.attribute (.name "s") "isdigit") [])
      heap (.scalar (.bool true)) heap := by
    simpa only [digits] using Evaluates.is_digit (module := program)
      (Evaluates.lookup (heap := heap) getS)
      (by intro name equal; cases equal; simp [getS])
  have negated : Evaluates program env
      (.unaryOp .not (.call (.attribute (.name "s") "isdigit") []))
      heap (.scalar (.bool false)) heap := Evaluates.unary tested rfl
  have condition : Evaluates program env invalidTest heap (.scalar (.bool false)) heap :=
    Evaluates.boolean_next compared rfl rfl (Evaluates.boolean_last negated rfl)
  exact BlockContinues.branch condition rfl .nil

theorem Represents.append (roots : Roots) (state : Storage) (heap extra : Mutable.Heap)
    (represented : Represents roots state heap) : Represents roots state (heap ++ extra) := by
  have retained (index : Nat) (object : Mutable.Object) (found : heap[index]? = some object) :
      (heap ++ extra)[index]? = some object := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  exact { represented with
    nodesFound := retained _ _ represented.nodesFound
    adjacencyFound := retained _ _ represented.adjacencyFound
    degreesFound := retained _ _ represented.degreesFound
    neighborsFound := fun vertex member => retained _ _ (represented.neighborsFound vertex member) }

abbrev character (value : Char) : String := String.ofList [value]

def unpackEnv (env : HeapCalls.Env) (a b c : Char) : HeapCalls.Env :=
  assignNames env ["a", "b", "c"] [text (character a), text (character b), text (character c)]

def unpackHeap (heap : Mutable.Heap) (a b c : Char) : Mutable.Heap :=
  heap ++ [.tuple [text (character a), text (character b), text (character c)]]

theorem unpack_three (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String) (a b c : Char)
    (getS : env "s" = some (text s)) (characters : s.toList = [a, b, c]) :
    BlockContinues program env unpackBody heap (unpackEnv env a b c) (unpackHeap heap a b c) := by
  have read (index : Int) (value : Char)
      (found : FormalSW.Py.Builtin.Sequence.get? [a, b, c] index = some value) :
      Evaluates program env (.subscript (.name "s") (.constant (.int index)))
        heap (text (character value)) heap := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getS) Evaluates.constant
    simp [Mutable.getItem, text, Strings.get, scalarNumber, characters, found, character]
  have evaluated : Evaluates program env
      (.tuple [.subscript (.name "s") (.constant (.int 0)),
        .subscript (.name "s") (.constant (.int 1)),
        .subscript (.name "s") (.constant (.int 2))]) heap (.ref heap.length)
      (unpackHeap heap a b c) :=
    Evaluates.tuple (.cons (read 0 a rfl) (.cons (read 1 b rfl) (.cons (read 2 c rfl) .nil)))
      (by intro impossible; cases impossible)
  apply BlockContinues.unpack rfl evaluated
  · simp [Mutable.sequence, unpackHeap]
  · rfl

theorem unpack_bindings (roots : Roots) (env : HeapCalls.Env) (a b c : Char)
    (bindings : Bindings roots env) : Bindings roots (unpackEnv env a b c) := by
  constructor
  · simpa [unpackEnv, assignNames, setLocal] using bindings.nodes
  · simpa [unpackEnv, assignNames, setLocal] using bindings.adjacency
  · simpa [unpackEnv, assignNames, setLocal] using bindings.degrees

/-- One complete successful source iteration, without restricting the digit alphabet to ASCII. -/
theorem valid_attempt (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (represented : Represents roots state heap)
    (bindings : Bindings roots env) (s : String) (getS : env "s" = some (text s))
    (length : s.length = 3) (digits : Strings.isDigit s = true) :
    ∃ a b c after next, s.toList = [a, b, c] ∧
      BlockContinues program env attemptBody heap (unpackEnv env a b c) after ∧
      Represents roots next after ∧ Bindings roots (unpackEnv env a b c) ∧
      next.vertices = insertVertex (character c)
        (insertVertex (character b) (insertVertex (character a) state.vertices)) ∧
      next.edges = insertEdge (character b) (character c)
        (insertEdge (character a) (character b) state.edges) ∧
      ∀ caller, CallerPreserved caller roots state heap →
        CallerPreserved caller roots next after := by
  obtain ⟨a, b, c, characters⟩ := List.length_eq_three.mp
    (show s.toList.length = 3 by simpa using length)
  have guard := valid_guard env heap s getS length digits
  have unpacked := unpack_three env heap s a b c getS characters
  have retained : Represents roots state (unpackHeap heap a b c) :=
    represented.append roots state heap _
  have bound := unpack_bindings roots env a b c bindings
  obtain ⟨after, next, updated, result, vertices, edges, saved⟩ :=
    updates_preserve roots state (unpackHeap heap a b c) (unpackEnv env a b c) retained bound
      (character a) (character b) (character c)
      (by simp [unpackEnv, assignNames, setLocal])
      (by simp [unpackEnv, assignNames, setLocal])
      (by simp [unpackEnv, assignNames, setLocal])
  refine ⟨a, b, c, after, next, characters, ?_, result, bound, vertices, edges, ?_⟩
  · rw [attempt_parts]
    exact (guard.append unpacked).append updated
  · intro caller before
    exact saved caller (before.append caller roots state heap _)

end solvers.«79».graph
