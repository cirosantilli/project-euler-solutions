import ProjectEulerSolutions.Lib.MonopolyOutputKeys
import lang.py.FormalSW.GeneratorCalls

namespace ProjectEulerSolutions.MonopolyOutput
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def formatSquare (square : Fin 40) : String := Strings.zfill (toString (square.val : Int)) 2

def formatExpression (parameter : String) : Expr :=
  .call (.attribute (.builtinCall .str [.name parameter]) "zfill") [.constant (.int 2)]

theorem format_result (module : Module) (square : Fin 40) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (parameter : String)
    (getSquare : env parameter = some (.scalar (.int square.val))) :
    Evaluates module env (formatExpression parameter) heap (.scalar (.string (formatSquare square))) heap globals := by
  have text : Evaluates module env (.builtinCall .str [.name parameter]) heap
      (.scalar (.string (toString (square.val : Int)))) heap globals :=
    Evaluates.builtin (.cons (.lookup getSquare) .nil) rfl
  exact Evaluates.zfill text (.cons (Evaluates.constant (value := .int 2)) .nil) rfl (by intros; contradiction)

theorem collect_strings (module : Module) (base : Mutable.Heap) (address : Nat)
    (squares remaining : List (Fin 40))
    (found : base[address]? = some (.list (squares.map fun square => .scalar (.int square.val))))
    (position : Nat) (suffix : squares.drop position = remaining) (output : List String)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (parameter : String) :
    ∃ finalEnv, BlockContinues module env
      [.sequenceStepLoop parameter (.ref address) position [.comprehensionAppend base.length (formatExpression parameter)] []]
      (base ++ [.list (output.map fun text => .scalar (.string text))]) finalEnv
      (base ++ [.list ((output ++ remaining.map formatSquare).map fun text => .scalar (.string text))]) globals := by
  have sequence (output : List String) :
      Mutable.sequence (.ref address) (base ++ [.list (output.map fun text => .scalar (.string text))]) =
        some (.ok (squares.map fun square => .scalar (.int square.val)),
          base ++ [.list (output.map fun text => .scalar (.string text))]) := by
    simp [Mutable.sequence, HeapCollection.prefix_found base _ address _ found]
  induction remaining generalizing position output env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.map_nil, List.append_nil]
      apply BlockContinues.sequence_done (source := .ref address) (sequence output) ?_ .nil
      simpa [List.getElem?_map, suffix] using
        congrArg (Option.map fun square : Fin 40 => Mutable.Value.scalar (.int square.val))
          (List.head?_drop (l := squares) (i := position)).symm
  | cons square rest ih =>
      let ready := setLocal env parameter (.scalar (.int square.val))
      have formatted := format_result module square
        (base ++ [.list (output.map fun text => .scalar (.string text))]) ready globals parameter (by simp [ready, setLocal])
      have emitted := BlockContinues.comprehension_append (address := base.length)
        (items := output.map fun text => .scalar (.string text)) formatted (by simp)
      obtain ⟨finalEnv, continued⟩ := ih (position + 1) (by simp [List.drop_add_one_eq_tail_drop, suffix])
        (output ++ [formatSquare square]) ready
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (source := .ref address) (value := .scalar (.int square.val)) (sequence output)
      · simpa [List.getElem?_map, suffix] using
          congrArg (Option.map fun square : Fin 40 => Mutable.Value.scalar (.int square.val))
            (List.head?_drop (l := squares) (i := position)).symm
      · have wrote : BlockContinues module ready [.comprehensionAppend base.length (formatExpression parameter)]
            (base ++ [.list (output.map fun text => .scalar (.string text))]) ready
            (base ++ [.list ((output ++ [formatSquare square]).map fun text => .scalar (.string text))]) globals := by
          simpa [List.set_append_right] using emitted
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or,
          Bool.false_eq_true, if_false, List.map_cons, List.append_assoc, List.singleton_append] using wrote.append continued

def topThreeExpression (parameter : String) : Expr :=
  .subscript (sortExpression parameter) (.slice none (some (.constant (.int 3))) none)

theorem top_three_result (module : Module) (values : Fin 120 → Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode (Monopoly.squareWeight values))))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (parameter : String) (different : parameter ≠ "square_weights")
    (getWeights : env "square_weights" = some (.ref address)) :
    ∃ extra result, Evaluates module env (topThreeExpression parameter) heap (.ref result) (heap ++ extra) globals ∧
      (heap ++ extra)[result]? = some (.list (((sortedSquares values).take 3).map fun square => .scalar (.int square.val))) := by
  obtain ⟨extra, result, sorted, sortedFound⟩ := sort_result module values heap address found env globals parameter different getWeights
  have size : (sortedSquares values).length = 40 := by
    simpa only [sortedSquares, List.length_finRange] using (HeapSort.sortBy_perm (less values) (List.finRange 40)).length_eq
  have operation := Mutable.getSlice_list_prefix (heap ++ extra) result _ 3 sortedFound
  have prefixSize : FormalSW.Py.Builtin.Slice.prefixLength
      ((sortedSquares values).map fun square => Mutable.Value.scalar (.int square.val)).length 3 = 3 := by
    rw [List.length_map, size]
    rfl
  rw [prefixSize, ← List.map_take] at operation
  refine ⟨extra ++ [.list (((sortedSquares values).take 3).map fun square => .scalar (.int square.val))],
    (heap ++ extra).length, ?_, ?_⟩
  · simpa only [List.append_assoc] using
      Evaluates.slice (.cons sorted (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil)))) operation
  · simp [← List.append_assoc]

def answer (values : Fin 120 → Int) : String :=
  String.intercalate "" (((sortedSquares values).take 3).map formatSquare)

def outputExpression (parameter : String) : Expr := .consumeGenerator
  (.attribute (.constant (.string "")) "join") (formatExpression parameter)
  [(.name parameter, topThreeExpression parameter, [])]

/-- Joining an arbitrary board-position list formats each yielded item exactly once. -/
theorem join_result (module : Module) (squares : List (Fin 40)) (heap middle : Mutable.Heap) (address : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (parameter : String)
    (evaluated : Evaluates module env (topThreeExpression parameter) heap (.ref address) middle globals)
    (found : middle[address]? = some (.list (squares.map fun square => .scalar (.int square.val)))) :
    Evaluates module env (outputExpression parameter) heap
      (.scalar (.string (String.intercalate "" (squares.map formatSquare))))
      (middle ++ [.list ((squares.map formatSquare).map fun text => .scalar (.string text))]) globals := by
  obtain ⟨finalEnv, collected⟩ := collect_strings module middle address squares squares found 0 rfl []
    (comprehensionTargetsEnv env [(.name parameter, topThreeExpression parameter, [])]) globals parameter
  apply Evaluates.consume_generator_join Evaluates.constant
  apply GeneratorCollects.sequence (target := .name parameter) (iterable := topThreeExpression parameter)
    (filters := []) (generators := []) (source := .ref address)
    (items := squares.map fun square => .scalar (.int square.val)) (middle := middle) (finalEnv := finalEnv)
    (collectedHeap := middle ++ [.list ((squares.map formatSquare).map fun text => .scalar (.string text))])
    (by intros; simp [topThreeExpression]) (by intros; simp [topThreeExpression])
    (by intros; simp [topThreeExpression]) evaluated (by simp [Mutable.sequence, found])
  · simpa only [List.map_nil, List.nil_append, comprehensionTargetName, comprehensionTargetBody,
      comprehensionFilters, comprehensionTargetsBody] using collected
  · exact primitive_join_list _ middle.length "" (squares.map formatSquare) (by simp)

/-- The complete return expression sorts, slices, formats and joins the same three square numbers. -/
theorem output_result (module : Module) (values : Fin 120 → Int) (heap : Mutable.Heap) (address : Nat)
    (found : heap[address]? = some (.list (encode (Monopoly.squareWeight values))))
    (env : HeapCalls.Env) (globals : Mutable.Globals) (parameter : String) (different : parameter ≠ "square_weights")
    (getWeights : env "square_weights" = some (.ref address)) :
    ∃ extra, Evaluates module env (outputExpression parameter) heap (.scalar (.string (answer values))) (heap ++ extra) globals := by
  obtain ⟨extra, result, sliced, sliceFound⟩ := top_three_result module values heap address found env globals parameter different getWeights
  refine ⟨extra ++ [.list ((((sortedSquares values).take 3).map formatSquare).map fun text => .scalar (.string text))], ?_⟩
  simpa only [answer, List.append_assoc] using join_result module ((sortedSquares values).take 3) heap (heap ++ extra)
    result env globals parameter sliced sliceFound

theorem answer_equal {sides : Nat} (positive : 0 < sides) {left right : Fin 120 → Int}
    (validLeft : Monopoly.IntegerWeights sides left) (validRight : Monopoly.IntegerWeights sides right) :
    answer left = answer right := by
  unfold answer sortedSquares
  rw [sort_equal positive validLeft validRight]

end ProjectEulerSolutions.MonopolyOutput
