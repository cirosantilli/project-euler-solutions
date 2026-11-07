import solvers.«98.py.lean.d».BijectionLoop
import data.«project-euler-statements».data.py.project_euler_statements.«p98.py.lean.d».Substitution

namespace solvers.«98».mapped_word
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.AnagramSquares
open pattern_signature (program)
open data.«project-euler-statements».data.py.project_euler_statements.p98.mapping
  (character value table lookup_assignment singleton_eq join_characters)
set_option Elab.async false

abbrev body : List Stmt := match (bijection.candidateBody[6]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []

abbrev element : Expr := .subscript (.name "char_to_digit") (.name "ch")
abbrev generators : List (Expr × Expr × List Expr) := [(.name "ch", .name "w2", [])]
abbrev joined : Expr := .consumeGenerator (.attribute (.constant (.string "")) "join") element generators
abbrev leadingZero : Expr := .compare
  (.subscript (.name "char_to_digit") (.subscript (.name "w2") (.constant (.int 0)))) [(.eq, .constant (.string "0"))]

theorem leading_zero_result (word digits : List Char) (other : String) (first : Char) (rest : List Char)
    (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (matched : labels [] word = labels [] digits) (member : first ∈ word) (view : other.toList = first :: rest)
    (found : heap[address]? = some (.dict (table (alphabet [] word) (alphabet [] digits))))
    (getMapping : env "char_to_digit" = some (.ref address)) (getOther : env "w2" = some (.scalar (.string other))) :
    Evaluates program env leadingZero heap (.scalar (.bool (decide (substitution '0' word digits first = '0')))) heap globals := by
  apply Evaluates.compare (av := value (substitution '0' word digits first)) (bv := value '0')
    (middle := heap) (afterRight := heap)
  · apply Evaluates.subscript (position := value first) (afterIndex := heap) (by intros; simp) (Evaluates.lookup getMapping)
    · apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getOther) Evaluates.constant
      simp [Mutable.getItem, Strings.get, scalarNumber, view,
        FormalSW.Py.Builtin.Sequence.get?, FormalSW.Py.Builtin.Sequence.normalizeIndex,
        show ¬(rest.length : Int) + 1 ≤ 0 by omega]
    · simp [Mutable.getItem, found, lookup_assignment word digits first matched member]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare]
    exact singleton_eq _ '0'

theorem collect (word digits remaining before : List Char) (other : String) (output : List Char)
    (heap : Mutable.Heap) (address : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (matched : labels [] word = labels [] digits) (included : ∀ letter ∈ remaining, letter ∈ word)
    (view : other.toList = before ++ remaining)
    (found : heap[address]? = some (.dict (table (alphabet [] word) (alphabet [] digits))))
    (getMapping : env "char_to_digit" = some (.ref address)) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "ch" (.scalar (.string other)) before.length [.comprehensionAppend heap.length element] []]
      (HeapCollection.working heap (output.map value) []) finalEnv
      (HeapCollection.working heap ((output ++ remaining.map (substitution '0' word digits)).map value) []) globals := by
  induction remaining generalizing before output env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.map_nil, List.append_nil]
      apply BlockContinues.sequence_done (items := other.toList.map value) rfl
      · simp [view]
      · exact BlockContinues.nil
  | cons letter rest ih =>
      let ready := setLocal env "ch" (value letter)
      have mapped : Evaluates program ready element (HeapCollection.working heap (output.map value) [])
          (value (substitution '0' word digits letter)) (HeapCollection.working heap (output.map value) []) globals := by
        apply Evaluates.subscript (by intros; simp)
          (Evaluates.lookup (show ready "char_to_digit" = some (.ref address) by simpa [ready, setLocal] using getMapping))
          (Evaluates.lookup (show ready "ch" = some (value letter) by simp [ready, setLocal]))
        have kept := HeapCollection.prefix_found heap [.list (output.map value)] address _ found
        simp only [HeapCollection.working] at kept ⊢
        simp [Mutable.getItem, kept, lookup_assignment word digits letter matched (included letter (by simp))]
      have emitted := BlockContinues.comprehension_append mapped (HeapCollection.found heap (output.map value) [])
      obtain ⟨finalEnv, continued⟩ := ih (before ++ [letter]) (output ++ [substitution '0' word digits letter]) ready
        (fun c member => included c (by simp [member]))
        (by simpa only [List.append_assoc, List.singleton_append] using view)
        (by simpa [ready, setLocal] using getMapping)
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (items := other.toList.map value) (value := value letter) rfl
      · simp [view]
      · have wrote : BlockContinues program ready [.comprehensionAppend heap.length element]
            (HeapCollection.working heap (output.map value) []) ready
            (HeapCollection.working heap ((output ++ [substitution '0' word digits letter]).map value) []) globals := by
          simpa only [HeapCollection.store, List.map_append, List.map_cons, List.map_nil] using emitted
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Bool.false_or,
          Bool.false_eq_true, if_false, List.length_append, List.length_singleton, List.map_cons,
          List.append_assoc, List.singleton_append] using wrote.append continued

/-- The generator performs every dictionary lookup and joins exactly the substituted anagram characters. -/
theorem joined_result (word digits : List Char) (other : String) (heap : Mutable.Heap) (address : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (matched : labels [] word = labels [] digits) (included : ∀ letter ∈ other.toList, letter ∈ word)
    (found : heap[address]? = some (.dict (table (alphabet [] word) (alphabet [] digits))))
    (getMapping : env "char_to_digit" = some (.ref address)) (getOther : env "w2" = some (.scalar (.string other))) :
    Evaluates program env joined heap (.scalar (.string (String.ofList (other.toList.map (substitution '0' word digits)))))
      (heap ++ [.list ((other.toList.map (substitution '0' word digits)).map value)]) globals := by
  let inside := comprehensionTargetsEnv env generators
  obtain ⟨finalEnv, collected⟩ := collect word digits other.toList [] other [] heap address inside globals matched included rfl found
    (by simpa [inside, comprehensionTargetsEnv, generators, targetLocals] using getMapping)
  apply Evaluates.consume_generator_join Evaluates.constant
  apply GeneratorCollects.sequence (target := .name "ch") (iterable := .name "w2")
    (filters := []) (generators := []) (source := .scalar (.string other)) (items := other.toList.map value)
    (middle := heap) (finalEnv := finalEnv)
    (collectedHeap := heap ++ [.list ((other.toList.map (substitution '0' word digits)).map value)])
    (by intros; simp) (by intros; simp) (by intros; simp) (Evaluates.lookup getOther) rfl
  · simpa only [HeapCollection.working, List.nil_append, List.append_nil, List.map_nil] using collected
  · have finished := primitive_join_list
      (heap ++ [.list ((other.toList.map (substitution '0' word digits)).map value)]) heap.length ""
      ((other.toList.map (substitution '0' word digits)).map String.singleton) (by simp [List.map_map, value, character])
    simpa only [GeneratorFinishes, join_characters] using finished

theorem source_leading_zero : body[0]? = some (.ifThen leadingZero [.continueLoop] []) := rfl

theorem source_substitution : body[1]? = some (.assign [.name "num2"] joined) := rfl

end solvers.«98».mapped_word
