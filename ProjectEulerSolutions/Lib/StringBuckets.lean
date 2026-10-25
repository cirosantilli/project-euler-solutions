import lang.py.FormalSW.KeyDictionaryCalls
import ProjectEulerSolutions.Lib.HeapCollection
import Mathlib.Data.List.Nodup

/-! Insertion-ordered string dictionaries whose values are distinct, shared lists of naturals. -/
namespace ProjectEulerSolutions.StringBuckets
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

def number (n : Nat) : Mutable.Value := .scalar (.int n)
def numbers (items : List Nat) : List Mutable.Value := items.map number

structure Bucket where
  key : String
  address : Nat
  values : List Nat
deriving Repr, DecidableEq

def entry (bucket : Bucket) : Constant × Mutable.Value := (.string bucket.key, .ref bucket.address)
def entries (buckets : List Bucket) := buckets.map entry
def object (buckets : List Bucket) : Mutable.Object := .dict (entries buckets)
def find (buckets : List Bucket) (key : String) : Option Bucket :=
  buckets.find? (fun bucket => decide (bucket.key = key))
def chosen (buckets : List Bucket) (key : String) (fresh : Nat) : Nat :=
  ((find buckets key).map Bucket.address).getD fresh
def contents (buckets : List Bucket) (key : String) : List Nat :=
  ((find buckets key).map Bucket.values).getD []

/-- A hit extends the existing shared list; a miss records the fresh default list. -/
def push : List Bucket → String → Nat → Nat → List Bucket
  | [], key, address, value => [⟨key, address, [value]⟩]
  | bucket :: rest, key, address, value =>
      if bucket.key = key then { bucket with values := bucket.values ++ [value] } :: rest
      else bucket :: push rest key address value

theorem lookup_entries (buckets : List Bucket) (key : String) :
    ScalarDict.lookup (entries buckets) (.string key) = (find buckets key).map (fun bucket => .ref bucket.address) := by
  induction buckets with
  | nil => rfl
  | cons bucket rest ih =>
      by_cases same : bucket.key = key <;>
        simp_all [entries, entry, ScalarDict.lookup, scalarCompare, find]

theorem entries_push (buckets : List Bucket) (key : String) (address value : Nat) :
    entries (push buckets key address value) =
      ScalarDict.insert (entries buckets) (.string key) (.ref (chosen buckets key address)) := by
  induction buckets with
  | nil => rfl
  | cons bucket rest ih =>
      by_cases same : bucket.key = key
      · simp [push, same, entries, entry, ScalarDict.insert, scalarCompare, chosen, find]
      · simpa [push, same, entries, entry, ScalarDict.insert, scalarCompare, chosen, find] using ih

theorem entries_push_hit (buckets : List Bucket) (key : String) (address value : Nat)
    (bucket : Bucket) (found : find buckets key = some bucket) :
    entries (push buckets key address value) = entries buckets := by
  induction buckets with
  | nil => simp [find] at found
  | cons first rest ih =>
      by_cases same : first.key = key
      · simp [push, same, entries, entry]
      · have tail : find rest key = some bucket := by simpa [find, same] using found
        simpa only [push, if_neg same, entries, List.map_cons] using congrArg (List.cons (entry first)) (ih tail)

theorem contents_push (buckets : List Bucket) (key query : String) (address value : Nat) :
    contents (push buckets key address value) query =
      if query = key then contents buckets key ++ [value] else contents buckets query := by
  induction buckets with
  | nil => by_cases same : key = query <;> simp [push, contents, find, same, Ne.symm]
  | cons first rest ih =>
      by_cases updated : first.key = key <;> by_cases selected : first.key = query <;>
        by_cases same : query = key <;> simp_all [push, contents, find]

theorem push_miss (buckets : List Bucket) (key : String) (address value : Nat)
    (missing : find buckets key = none) :
    push buckets key address value = buckets ++ [⟨key, address, [value]⟩] := by
  induction buckets with
  | nil => rfl
  | cons first rest ih =>
      have different : first.key ≠ key := by
        intro same
        simp [find, same] at missing
      have tail : find rest key = none := by simpa [find, different] using missing
      simp [push, different, ih tail]

theorem push_split (before after : List Bucket) (selected : Bucket) (key : String)
    (matched : selected.key = key) (skipped : ∀ bucket ∈ before, bucket.key ≠ key) (address value : Nat) :
    push (before ++ selected :: after) key address value =
      before ++ { selected with values := selected.values ++ [value] } :: after := by
  induction before with
  | nil => simp [push, matched]
  | cons first rest ih =>
      have different := skipped first List.mem_cons_self
      simp only [List.cons_append, push, if_neg different]
      rw [ih (fun bucket member => skipped bucket (List.mem_cons_of_mem _ member))]

theorem addresses_push (buckets : List Bucket) (key : String) (address value : Nat) :
    (push buckets key address value).map Bucket.address =
      if (find buckets key).isSome then buckets.map Bucket.address else buckets.map Bucket.address ++ [address] := by
  induction buckets with
  | nil => rfl
  | cons first rest ih =>
      by_cases same : first.key = key
      · simp [push, same, find]
      · have skipped : find (first :: rest) key = find rest key := by simp [find, same]
        simp only [push, if_neg same, List.map_cons, ih, skipped]
        split <;> simp_all

theorem keys_push (buckets : List Bucket) (key : String) (address value : Nat) :
    (push buckets key address value).map Bucket.key =
      if (find buckets key).isSome then buckets.map Bucket.key else buckets.map Bucket.key ++ [key] := by
  induction buckets with
  | nil => rfl
  | cons first rest ih =>
      by_cases same : first.key = key
      · simp [push, same, find]
      · have skipped : find (first :: rest) key = find rest key := by simp [find, same]
        simp only [push, if_neg same, List.map_cons, ih, skipped]
        split <;> simp_all

theorem find_of_mem (buckets : List Bucket) (bucket : Bucket) (member : bucket ∈ buckets)
    (unique : (buckets.map Bucket.key).Nodup) : find buckets bucket.key = some bucket := by
  induction buckets with
  | nil => simp at member
  | cons first rest ih =>
      have distinct := List.nodup_cons.mp unique
      rcases List.mem_cons.mp member with rfl | member
      · simp [find]
      · have different : first.key ≠ bucket.key := by
          intro same
          apply distinct.1
          rw [same]
          exact List.mem_map_of_mem member
        simpa [find, different] using ih member distinct.2

/-- The heap addresses are independent of the exact mathematical partition of the input. -/
structure Partition (keyOf : Nat → String) (items : List Nat) (buckets : List Bucket) : Prop where
  unique : (buckets.map Bucket.key).Nodup
  contents : ∀ key, contents buckets key = items.filter (fun value => decide (keyOf value = key))

theorem Partition.empty (keyOf : Nat → String) : Partition keyOf [] [] := by
  constructor
  · simp
  · intro key
    rfl

theorem Partition.push {keyOf : Nat → String} {items : List Nat} {buckets : List Bucket}
    (partition : Partition keyOf items buckets) (address value : Nat) :
    Partition keyOf (items ++ [value]) (push buckets (keyOf value) address value) := by
  constructor
  · rw [keys_push]
    cases selected : find buckets (keyOf value) with
    | some bucket => simpa only [selected, Option.isSome_some, if_true] using partition.unique
    | none =>
        have missing : keyOf value ∉ buckets.map Bucket.key := by
          intro member
          obtain ⟨bucket, inside, same⟩ := List.mem_map.mp member
          have found := find_of_mem buckets bucket inside partition.unique
          rw [same, selected] at found
          cases found
        simpa only [selected, Option.isSome_none, Bool.false_eq_true, if_false, List.concat_eq_append] using
          List.Nodup.concat missing partition.unique
  · intro key
    rw [contents_push, partition.contents, partition.contents, List.filter_append]
    by_cases same : keyOf value = key <;> simp [same, Ne.symm]

theorem Partition.bucket_values {keyOf : Nat → String} {items : List Nat} {buckets : List Bucket}
    (partition : Partition keyOf items buckets) (bucket : Bucket) (member : bucket ∈ buckets) :
    bucket.values = items.filter (fun value => decide (keyOf value = bucket.key)) := by
  have selected := find_of_mem buckets bucket member partition.unique
  simpa only [StringBuckets.contents, selected, Option.map_some, Option.getD_some] using partition.contents bucket.key

/-- Every input value is present in the unique bucket selected by its key. -/
theorem Partition.member {keyOf : Nat → String} {items : List Nat} {buckets : List Bucket}
    (partition : Partition keyOf items buckets) (value : Nat) (member : value ∈ items) :
    ∃ bucket ∈ buckets, bucket.key = keyOf value ∧ value ∈ bucket.values := by
  have present : value ∈ StringBuckets.contents buckets (keyOf value) := by
    rw [partition.contents]
    simp [member]
  cases selected : find buckets (keyOf value) with
  | none => simp [StringBuckets.contents, selected] at present
  | some bucket =>
      refine ⟨bucket, List.mem_of_find?_eq_some selected, ?_, ?_⟩
      · exact of_decide_eq_true
          (List.find?_some (p := fun entry : Bucket => decide (entry.key = keyOf value)) selected)
      · simpa only [StringBuckets.contents, selected, Option.map_some, Option.getD_some] using present

structure View (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) : Prop where
  found : heap[dictionary]? = some (object buckets)
  lists : ∀ bucket ∈ buckets, heap[bucket.address]? = some (.list (numbers bucket.values))
  afterDictionary : ∀ bucket ∈ buckets, dictionary < bucket.address
  unique : (buckets.map Bucket.address).Nodup

theorem View.append {heap : Mutable.Heap} {dictionary : Nat} {buckets : List Bucket}
    (view : View heap dictionary buckets) (extra : Mutable.Heap) : View (heap ++ extra) dictionary buckets :=
  ⟨HeapCollection.prefix_found heap extra dictionary _ view.found,
    fun bucket member => HeapCollection.prefix_found heap extra bucket.address _ (view.lists bucket member),
    view.afterDictionary, view.unique⟩

theorem empty_view (heap : Mutable.Heap) : View (heap ++ [.dict []]) heap.length [] := by
  constructor
  · simp [object, entries]
  · simp
  · simp
  · simp

def prepared (heap : Mutable.Heap) : Mutable.Heap := heap ++ [.list []]
def pending (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (key : String) (value : Nat) : Mutable.Heap :=
  if (find buckets key).isSome then prepared heap
  else (prepared heap).set dictionary (object (push buckets key heap.length value))

/-- The default allocation precedes lookup, including when an existing bucket wins. -/
theorem setdefault (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (found : heap[dictionary]? = some (object buckets)) (key : String) (value : Nat) :
    Mutable.dictionarySetDefault dictionary (.scalar (.string key)) (.ref heap.length) (prepared heap) =
      some (.ok (.ref (chosen buckets key heap.length)), pending heap dictionary buckets key value) := by
  have original : (prepared heap)[dictionary]? = some (object buckets) :=
    HeapCollection.prefix_found heap _ dictionary _ found
  have operation := Mutable.dictionarySetDefault_scalar (prepared heap) dictionary (entries buckets) original
    (.string key) (.ref heap.length)
  cases selected : find buckets key with
  | none =>
      simpa only [lookup_entries, selected, Option.map_none, Option.getD_none, Option.isSome_none,
        Bool.false_eq_true, if_false, pending, chosen, object, entries_push] using operation
  | some bucket =>
      simpa only [lookup_entries, selected, Option.map_some, Option.getD_some, Option.isSome_some,
        if_true, pending, chosen] using operation

theorem selected_values (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : String) (value : Nat) :
    (pending heap dictionary buckets key value)[chosen buckets key heap.length]? =
      some (.list (numbers (contents buckets key))) := by
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      have different : dictionary ≠ heap.length := by omega
      simp only [pending, chosen, contents, selected, Option.isSome_none, Bool.false_eq_true,
        if_false, Option.map_none, Option.getD_none]
      rw [List.getElem?_set_ne different]
      simp [prepared, numbers]
  | some bucket =>
      simpa [pending, chosen, contents, selected] using
        HeapCollection.prefix_found heap [.list []] bucket.address _
          (view.lists bucket (List.mem_of_find?_eq_some selected))

def updated (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (key : String) (value : Nat) : Mutable.Heap :=
  (pending heap dictionary buckets key value).set (chosen buckets key heap.length)
    (.list (numbers (contents buckets key ++ [value])))

theorem chosen_after (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : String) : dictionary < chosen buckets key heap.length := by
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      simp only [chosen, selected, Option.map_none, Option.getD_none]
      omega
  | some bucket =>
      simpa only [chosen, selected, Option.map_some, Option.getD_some] using
        view.afterDictionary bucket (List.mem_of_find?_eq_some selected)

theorem updated_selected (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : String) (value : Nat) :
    (updated heap dictionary buckets key value)[chosen buckets key heap.length]? =
      some (.list (numbers (contents buckets key ++ [value]))) := by
  exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp (selected_values heap dictionary buckets view key value)).1

theorem updated_other (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : String) (value : Nat) (bucket : Bucket)
    (member : bucket ∈ buckets) (different : chosen buckets key heap.length ≠ bucket.address) :
    (updated heap dictionary buckets key value)[bucket.address]? = heap[bucket.address]? := by
  unfold updated
  rw [List.getElem?_set_ne different]
  have inside := (List.getElem?_eq_some_iff.mp (view.lists bucket member)).1
  have afterDictionary := view.afterDictionary bucket member
  unfold pending
  split
  · exact List.getElem?_append_left inside
  · rw [List.getElem?_set_ne (by omega : dictionary ≠ bucket.address)]
    exact List.getElem?_append_left inside

theorem updated_dictionary (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : String) (value : Nat) :
    (updated heap dictionary buckets key value)[dictionary]? =
      some (object (push buckets key heap.length value)) := by
  have different := chosen_after heap dictionary buckets view key
  unfold updated
  rw [List.getElem?_set_ne (by omega : chosen buckets key heap.length ≠ dictionary)]
  cases selected : find buckets key with
  | none =>
      have inside := (List.getElem?_eq_some_iff.mp view.found).1
      simp only [pending, selected, Option.isSome_none, Bool.false_eq_true, if_false]
      exact List.getElem?_set_self (by simp [prepared]; omega)
  | some bucket =>
      have same : object (push buckets key heap.length value) = object buckets := by
        rw [object, entries_push_hit buckets key _ _ bucket selected]
        rfl
      rw [same]
      simpa only [pending, selected, Option.isSome_some, if_true] using
        HeapCollection.prefix_found heap _ dictionary _ view.found

/-- The shared input objects before the dictionary survive every allocation and write. -/
theorem updated_prefix (base suffix : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View (base ++ suffix) dictionary buckets) (afterBase : base.length ≤ dictionary)
    (key : String) (value : Nat) :
    ∃ extra, updated (base ++ suffix) dictionary buckets key value = base ++ extra := by
  have afterChosen : base.length ≤ chosen buckets key (base ++ suffix).length := by
    have later := chosen_after (base ++ suffix) dictionary buckets view key
    omega
  have preparedPrefix : prepared (base ++ suffix) = base ++ (suffix ++ [.list []]) := by
    simp [prepared, List.append_assoc]
  unfold updated pending
  split
  · rw [preparedPrefix, List.set_append_right _ _ afterChosen]
    exact ⟨_, rfl⟩
  · rw [preparedPrefix, List.set_append_right _ _ afterBase, List.set_append_right _ _ afterChosen]
    exact ⟨_, rfl⟩

/-- Insertion preserves all existing bucket identities and keeps their lists disjoint. -/
theorem updated_view (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket)
    (view : View heap dictionary buckets) (key : String) (value : Nat) :
    View (updated heap dictionary buckets key value) dictionary (push buckets key heap.length value) := by
  have fresh : heap.length ∉ buckets.map Bucket.address := by
    intro member
    obtain ⟨bucket, inside, equal⟩ := List.mem_map.mp member
    have bound := (List.getElem?_eq_some_iff.mp (view.lists bucket inside)).1
    omega
  constructor
  · exact updated_dictionary heap dictionary buckets view key value
  · intro current member
    cases selected : find buckets key with
    | none =>
        rw [push_miss buckets key _ _ selected] at member
        rcases List.mem_append.mp member with old | added
        · have bound := (List.getElem?_eq_some_iff.mp (view.lists current old)).1
          have different : chosen buckets key heap.length ≠ current.address := by
            simp only [chosen, selected, Option.map_none, Option.getD_none]
            omega
          rw [updated_other heap dictionary buckets view key value current old different]
          exact view.lists current old
        · obtain rfl := List.mem_singleton.mp added
          simpa only [chosen, contents, selected, Option.map_none, Option.getD_none, List.nil_append] using
            updated_selected heap dictionary buckets view key value
    | some bucket =>
        obtain ⟨matched, before, after, split, skipped⟩ := List.find?_eq_some_iff_append.mp selected
        have keyEqual : bucket.key = key := of_decide_eq_true matched
        have skip : ∀ entry ∈ before, entry.key ≠ key := by
          intro entry inside
          simpa using skipped entry inside
        have distinct : (before.map Bucket.address ++ bucket.address :: after.map Bucket.address).Nodup := by
          simpa only [split, List.map_append, List.map_cons] using view.unique
        have moved : (bucket.address :: (before.map Bucket.address ++ after.map Bucket.address)).Nodup :=
          List.perm_middle.nodup_iff.mp distinct
        have without := (List.nodup_cons.mp moved).1
        have kept (entry : Bucket) (inside : entry ∈ before ++ after) :
            (updated heap dictionary buckets key value)[entry.address]? = some (.list (numbers entry.values)) := by
          have old : entry ∈ buckets := by
            rw [split]
            rcases List.mem_append.mp inside with left | right
            · exact List.mem_append.mpr (.inl left)
            · exact List.mem_append.mpr (.inr (List.mem_cons_of_mem _ right))
          have different : chosen buckets key heap.length ≠ entry.address := by
            simp only [chosen, selected, Option.map_some, Option.getD_some]
            intro equal
            apply without
            rw [equal]
            simpa only [List.map_append] using List.mem_map_of_mem (f := Bucket.address) inside
          rw [updated_other heap dictionary buckets view key value entry old different]
          exact view.lists entry old
        rw [split, push_split before after bucket key keyEqual skip] at member
        rcases List.mem_append.mp member with left | right
        · exact kept current (List.mem_append.mpr (.inl left))
        · rcases List.mem_cons.mp right with rfl | right
          · simpa only [chosen, contents, selected, Option.map_some, Option.getD_some] using
              updated_selected heap dictionary buckets view key value
          · exact kept current (List.mem_append.mpr (.inr right))
  · intro current member
    have addressMember := List.mem_map_of_mem (f := Bucket.address) member
    rw [addresses_push] at addressMember
    cases selected : find buckets key with
    | none =>
        simp only [selected, Option.isSome_none, Bool.false_eq_true, if_false] at addressMember
        rcases List.mem_append.mp addressMember with old | added
        · obtain ⟨entry, inside, equal⟩ := List.mem_map.mp old
          rw [← equal]
          exact view.afterDictionary entry inside
        · have equal := List.mem_singleton.mp added
          have bound := (List.getElem?_eq_some_iff.mp view.found).1
          omega
    | some bucket =>
        simp only [selected, Option.isSome_some, if_true] at addressMember
        obtain ⟨entry, inside, equal⟩ := List.mem_map.mp addressMember
        rw [← equal]
        exact view.afterDictionary entry inside
  · rw [addresses_push]
    split
    · exact view.unique
    · simpa only [List.concat_eq_append] using List.Nodup.concat fresh view.unique

def insertion : Expr := .call
  (.attribute (.call (.attribute (.name "groups") "setdefault") [.name "key", .list []]) "append") [.name "p"]

/-- Execute the unchanged setdefault/append expression, including the discarded default on a hit. -/
theorem evaluate {globals : Mutable.Globals} (program : Module) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (dictionary : Nat) (buckets : List Bucket) (view : View heap dictionary buckets)
    (key : String) (value : Nat) (getMap : env "groups" = some (.ref dictionary))
    (getKey : env "key" = some (.scalar (.string key))) (getP : env "p" = some (number value)) :
    Evaluates (globals := globals) program env insertion heap (.scalar .none)
      (updated heap dictionary buckets key value) := by
  have bound : bindMethod program (.ref dictionary) "setdefault" heap =
      some (.ok (.dictionarySetDefault dictionary), heap) := by
    simp [bindMethod, view.found, object]
  have receiver : Evaluates (globals := globals) program env
      (.call (.attribute (.name "groups") "setdefault") [.name "key", .list []])
      heap (.ref (chosen buckets key heap.length)) (pending heap dictionary buckets key value) := by
    apply Evaluates.dictionary_setdefault
      (by intro name same; cases same; simp [getMap]) (Evaluates.lookup getMap) bound
      (.cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil))
    exact setdefault heap dictionary buckets view.found key value
  have selected := selected_values heap dictionary buckets view key value
  simpa only [insertion, updated, numbers, List.map_append, List.map_cons, List.map_nil] using
    Evaluates.append receiver selected (Evaluates.lookup getP) selected

end ProjectEulerSolutions.StringBuckets
