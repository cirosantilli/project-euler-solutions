import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8467_small : List Nat := []
theorem row8467_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8467 row8467_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8467_block_005 : List Nat := [8689, 9151, 9697, 10069]
def plan8467_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), (.secondFactor 17), .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 13), .second, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 19), (.secondFactor 29), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .second, .first, .first, .first, (.firstFactor 31), (.firstFactor 37), (.secondFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .second, (.firstFactor 29), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row8467_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8467 row8467_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8467_block_005
  · decide
  · rfl

def row8467_block_006 : List Nat := [10729, 10903]
def plan8467_block_006 : List Budget.Choice := [.first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .possible, (.firstFactor 11), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 29), .second, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row8467_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8467 row8467_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8467_block_006
  · decide
  · rfl

def row8467_block_007 : List Nat := [12583, 13597, 13669]
def plan8467_block_007 : List Budget.Choice := [(.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.firstFactor 23), .second, .first, (.secondFactor 23), .possible, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 37), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, .possible, .first, (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.secondFactor 17), .second, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, (.firstFactor 11), (.secondFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8467_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8467 row8467_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8467_block_007
  · decide
  · rfl

def row8467_block_008 : List Nat := [14983, 15607, 15937]
def plan8467_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 13), .first, .first, .first, .possible, .first, .first, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 19), .possible, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .second, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8467_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8467 row8467_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8467_block_008
  · decide
  · rfl

def row8467_block_009 : List Nat := []
def plan8467_block_009 : List Budget.Choice := [(.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), (.secondFactor 11), (.secondFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), (.secondFactor 11), .second, .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8467_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8467 row8467_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8467_block_009
  · decide
  · rfl

def row8467_block_010 : List Nat := []
theorem row8467_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8467 row8467_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8467_block_011 : List Nat := []
theorem row8467_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8467 row8467_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8467_block_012 : List Nat := []
theorem row8467_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8467 row8467_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8467_block_013 : List Nat := []
theorem row8467_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8467 row8467_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8467_large_left_left : List Nat := row8467_small ++ row8467_block_005
theorem row8467_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8467 row8467_large_left_left :=
  Budget.RowUpperBound.branch row8467_small_covers row8467_block_005_covers

def row8467_large_left_right_right : List Nat := row8467_block_007 ++ row8467_block_008
theorem row8467_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8467 row8467_large_left_right_right :=
  Budget.RowUpperBound.branch row8467_block_007_covers row8467_block_008_covers

def row8467_large_left_right : List Nat := row8467_block_006 ++ row8467_large_left_right_right
theorem row8467_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8467 row8467_large_left_right :=
  Budget.RowUpperBound.branch row8467_block_006_covers row8467_large_left_right_right_covers

def row8467_large_left : List Nat := row8467_large_left_left ++ row8467_large_left_right
theorem row8467_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8467 row8467_large_left :=
  Budget.RowUpperBound.branch row8467_large_left_left_covers row8467_large_left_right_covers

def row8467_large_right_left : List Nat := row8467_block_009 ++ row8467_block_010
theorem row8467_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8467 row8467_large_right_left :=
  Budget.RowUpperBound.branch row8467_block_009_covers row8467_block_010_covers

def row8467_large_right_right_right : List Nat := row8467_block_012 ++ row8467_block_013
theorem row8467_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8467 row8467_large_right_right_right :=
  Budget.RowUpperBound.branch row8467_block_012_covers row8467_block_013_covers

def row8467_large_right_right : List Nat := row8467_block_011 ++ row8467_large_right_right_right
theorem row8467_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8467 row8467_large_right_right :=
  Budget.RowUpperBound.branch row8467_block_011_covers row8467_large_right_right_right_covers

def row8467_large_right : List Nat := row8467_large_right_left ++ row8467_large_right_right
theorem row8467_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8467 row8467_large_right :=
  Budget.RowUpperBound.branch row8467_large_right_left_covers row8467_large_right_right_covers

def row8467_large : List Nat := row8467_large_left ++ row8467_large_right
theorem row8467_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8467 row8467_large :=
  Budget.RowUpperBound.branch row8467_large_left_covers row8467_large_right_covers

def row8467 : List Nat := row8467_large
theorem row8467_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8467 row8467 := row8467_large_covers

def row8501_small : List Nat := []
theorem row8501_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8501 row8501_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8501_block_005 : List Nat := [8951, 9719]
def plan8501_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .second, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), (.secondFactor 29), .first, .first, (.secondFactor 19), .first, (.firstFactor 29), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 37), .first, .possible, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 17), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 23), (.firstFactor 17), (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 11), .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .second, .first, (.firstFactor 7), (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8501_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8501 row8501_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8501_block_005
  · decide
  · rfl

def row8501_block_006 : List Nat := [12401]
def plan8501_block_006 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .second, (.secondFactor 19), (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .second, (.firstFactor 37), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .second, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 23), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 23), .second, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .second, .first, .first, (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 29), (.secondFactor 31), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 17), .second, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, .first, .second]
set_option maxRecDepth 2048 in
theorem row8501_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8501 row8501_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8501_block_006
  · decide
  · rfl

def row8501_block_007 : List Nat := [13463, 13613, 14081]
def plan8501_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), (.firstFactor 19), (.firstFactor 23), .first, .first, .first, (.secondFactor 13), (.secondFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 19), .possible, (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, (.secondFactor 11), .first, (.secondFactor 13), (.firstFactor 7), .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, .possible, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .second, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8501_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8501 row8501_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8501_block_007
  · decide
  · rfl

def row8501_block_008 : List Nat := [14831, 15131, 15467, 15641, 16067, 16097, 16217, 16349]
def plan8501_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 13), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 37), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), (.firstFactor 23), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .possible, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 17), .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 19), (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 29), .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 11), .possible, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 31), .possible, (.firstFactor 11), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 11), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row8501_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8501 row8501_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8501_block_008
  · decide
  · rfl

def row8501_block_009 : List Nat := []
def plan8501_block_009 : List Budget.Choice := [.first, .second, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 13), .second, (.firstFactor 23), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 31), (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 13), (.firstFactor 31), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, .first, .first, .second, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .second, (.firstFactor 13), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8501_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8501 row8501_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8501_block_009
  · decide
  · rfl

def row8501_block_010 : List Nat := []
theorem row8501_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8501 row8501_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8501_block_011 : List Nat := []
theorem row8501_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8501 row8501_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8501_block_012 : List Nat := []
theorem row8501_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8501 row8501_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8501_block_013 : List Nat := []
theorem row8501_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8501 row8501_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8501_large_left_left : List Nat := row8501_small ++ row8501_block_005
theorem row8501_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8501 row8501_large_left_left :=
  Budget.RowUpperBound.branch row8501_small_covers row8501_block_005_covers

def row8501_large_left_right_right : List Nat := row8501_block_007 ++ row8501_block_008
theorem row8501_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8501 row8501_large_left_right_right :=
  Budget.RowUpperBound.branch row8501_block_007_covers row8501_block_008_covers

def row8501_large_left_right : List Nat := row8501_block_006 ++ row8501_large_left_right_right
theorem row8501_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8501 row8501_large_left_right :=
  Budget.RowUpperBound.branch row8501_block_006_covers row8501_large_left_right_right_covers

def row8501_large_left : List Nat := row8501_large_left_left ++ row8501_large_left_right
theorem row8501_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8501 row8501_large_left :=
  Budget.RowUpperBound.branch row8501_large_left_left_covers row8501_large_left_right_covers

def row8501_large_right_left : List Nat := row8501_block_009 ++ row8501_block_010
theorem row8501_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8501 row8501_large_right_left :=
  Budget.RowUpperBound.branch row8501_block_009_covers row8501_block_010_covers

def row8501_large_right_right_right : List Nat := row8501_block_012 ++ row8501_block_013
theorem row8501_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8501 row8501_large_right_right_right :=
  Budget.RowUpperBound.branch row8501_block_012_covers row8501_block_013_covers

def row8501_large_right_right : List Nat := row8501_block_011 ++ row8501_large_right_right_right
theorem row8501_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8501 row8501_large_right_right :=
  Budget.RowUpperBound.branch row8501_block_011_covers row8501_large_right_right_right_covers

def row8501_large_right : List Nat := row8501_large_right_left ++ row8501_large_right_right
theorem row8501_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8501 row8501_large_right :=
  Budget.RowUpperBound.branch row8501_large_right_left_covers row8501_large_right_right_covers

def row8501_large : List Nat := row8501_large_left ++ row8501_large_right
theorem row8501_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8501 row8501_large :=
  Budget.RowUpperBound.branch row8501_large_left_covers row8501_large_right_covers

def row8501 : List Nat := row8501_large
theorem row8501_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8501 row8501 := row8501_large_covers

def row8513_small : List Nat := []
theorem row8513_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8513 row8513_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8513_block_005 : List Nat := [8747, 8831, 9227, 10007, 10289, 10331]
def plan8513_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, .possible, (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .possible, .second, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .second, (.secondFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, (.firstFactor 37), (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 13), .second, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .possible, (.secondFactor 13), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row8513_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8513 row8513_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8513_block_005
  · decide
  · rfl

def row8513_block_006 : List Nat := [10709, 11399, 11831]
def plan8513_block_006 : List Budget.Choice := [(.secondFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 11), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 17), .first, .second, (.firstFactor 19), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .second, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 31), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 31), (.secondFactor 7), (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 31), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, .first, .second, (.firstFactor 7), (.firstFactor 11), .second, .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, (.secondFactor 13), .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8513_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8513 row8513_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8513_block_006
  · decide
  · rfl

def row8513_block_007 : List Nat := [12821, 13163, 13469, 13901]
def plan8513_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), (.secondFactor 29), (.secondFactor 31), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .possible, .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .possible, .first, .first, .first, (.secondFactor 13), (.secondFactor 31), .first, (.firstFactor 13), (.firstFactor 29), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 7), .possible, .first, (.firstFactor 19), (.secondFactor 13), .first, (.secondFactor 29), .first, (.secondFactor 11), .first, (.secondFactor 13), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .second, .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .possible, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8513_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8513 row8513_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8513_block_007
  · decide
  · rfl

def row8513_block_008 : List Nat := [14699, 15053, 15299, 15791]
def plan8513_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 11), .second, .first, .first, .second, .first, (.firstFactor 31), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .possible, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 37), .first, .second, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 23), (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 31), .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 29), (.firstFactor 7), .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 17), (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .possible, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, .first, (.secondFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 37), .second, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 19), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.secondFactor 29), (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row8513_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8513 row8513_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8513_block_008
  · decide
  · rfl

def row8513_block_009 : List Nat := [17189, 17333, 17387]
def plan8513_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.firstFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 11), .second, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .possible, .first, (.secondFactor 7), .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8513_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8513 row8513_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8513_block_009
  · decide
  · rfl

def row8513_block_010 : List Nat := []
theorem row8513_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8513 row8513_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8513_block_011 : List Nat := []
theorem row8513_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8513 row8513_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8513_block_012 : List Nat := []
theorem row8513_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8513 row8513_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8513_block_013 : List Nat := []
theorem row8513_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8513 row8513_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8513_large_left_left : List Nat := row8513_small ++ row8513_block_005
theorem row8513_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8513 row8513_large_left_left :=
  Budget.RowUpperBound.branch row8513_small_covers row8513_block_005_covers

def row8513_large_left_right_right : List Nat := row8513_block_007 ++ row8513_block_008
theorem row8513_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8513 row8513_large_left_right_right :=
  Budget.RowUpperBound.branch row8513_block_007_covers row8513_block_008_covers

def row8513_large_left_right : List Nat := row8513_block_006 ++ row8513_large_left_right_right
theorem row8513_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8513 row8513_large_left_right :=
  Budget.RowUpperBound.branch row8513_block_006_covers row8513_large_left_right_right_covers

def row8513_large_left : List Nat := row8513_large_left_left ++ row8513_large_left_right
theorem row8513_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8513 row8513_large_left :=
  Budget.RowUpperBound.branch row8513_large_left_left_covers row8513_large_left_right_covers

def row8513_large_right_left : List Nat := row8513_block_009 ++ row8513_block_010
theorem row8513_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8513 row8513_large_right_left :=
  Budget.RowUpperBound.branch row8513_block_009_covers row8513_block_010_covers

def row8513_large_right_right_right : List Nat := row8513_block_012 ++ row8513_block_013
theorem row8513_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8513 row8513_large_right_right_right :=
  Budget.RowUpperBound.branch row8513_block_012_covers row8513_block_013_covers

def row8513_large_right_right : List Nat := row8513_block_011 ++ row8513_large_right_right_right
theorem row8513_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8513 row8513_large_right_right :=
  Budget.RowUpperBound.branch row8513_block_011_covers row8513_large_right_right_right_covers

def row8513_large_right : List Nat := row8513_large_right_left ++ row8513_large_right_right
theorem row8513_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8513 row8513_large_right :=
  Budget.RowUpperBound.branch row8513_large_right_left_covers row8513_large_right_right_covers

def row8513_large : List Nat := row8513_large_left ++ row8513_large_right
theorem row8513_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8513 row8513_large :=
  Budget.RowUpperBound.branch row8513_large_left_covers row8513_large_right_covers

def row8513 : List Nat := row8513_large
theorem row8513_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8513 row8513 := row8513_large_covers

def row8521_small : List Nat := []
theorem row8521_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8521 row8521_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8521_block_005 : List Nat := [8677, 8821, 8923, 9133, 9157, 9613, 9619, 10009]
def plan8521_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .second, (.secondFactor 23), .first, .second, .first, .first, .possible, .first, .first, (.secondFactor 17), .first, .first, .second, .first, (.firstFactor 13), (.secondFactor 7), .possible, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 37), (.secondFactor 19), (.firstFactor 7), .possible, .first, .first, .possible, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .second, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 19), .first, .second, .first, .second, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 37), .first, .first, (.firstFactor 23), .first, (.firstFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8521_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8521 row8521_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8521_block_005
  · decide
  · rfl

def row8521_block_006 : List Nat := [10837, 11311]
def plan8521_block_006 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .second, .first, .first, (.secondFactor 23), (.secondFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 37), .first, .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row8521_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8521 row8521_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8521_block_006
  · decide
  · rfl

def row8521_block_007 : List Nat := [12889, 13417]
def plan8521_block_007 : List Budget.Choice := [(.secondFactor 7), .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.secondFactor 11), (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 19), .second, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 31), .first, .possible, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, .first, .first, .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .possible, .first, (.firstFactor 37), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 13), .first, .second, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8521_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8521 row8521_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8521_block_007
  · decide
  · rfl

def row8521_block_008 : List Nat := [14503, 15139, 15241]
def plan8521_block_008 : List Budget.Choice := [(.secondFactor 7), .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.firstFactor 37), .first, .first, .possible, (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.secondFactor 29), (.firstFactor 7), .second, .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 17), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8521_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8521 row8521_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8521_block_008
  · decide
  · rfl

def row8521_block_009 : List Nat := []
def plan8521_block_009 : List Budget.Choice := [(.secondFactor 19), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 31), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 23), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 11), .second, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8521_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8521 row8521_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8521_block_009
  · decide
  · rfl

def row8521_block_010 : List Nat := []
theorem row8521_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8521 row8521_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8521_block_011 : List Nat := []
theorem row8521_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8521 row8521_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8521_block_012 : List Nat := []
theorem row8521_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8521 row8521_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8521_block_013 : List Nat := []
theorem row8521_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8521 row8521_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8521_large_left_left : List Nat := row8521_small ++ row8521_block_005
theorem row8521_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8521 row8521_large_left_left :=
  Budget.RowUpperBound.branch row8521_small_covers row8521_block_005_covers

def row8521_large_left_right_right : List Nat := row8521_block_007 ++ row8521_block_008
theorem row8521_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8521 row8521_large_left_right_right :=
  Budget.RowUpperBound.branch row8521_block_007_covers row8521_block_008_covers

def row8521_large_left_right : List Nat := row8521_block_006 ++ row8521_large_left_right_right
theorem row8521_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8521 row8521_large_left_right :=
  Budget.RowUpperBound.branch row8521_block_006_covers row8521_large_left_right_right_covers

def row8521_large_left : List Nat := row8521_large_left_left ++ row8521_large_left_right
theorem row8521_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8521 row8521_large_left :=
  Budget.RowUpperBound.branch row8521_large_left_left_covers row8521_large_left_right_covers

def row8521_large_right_left : List Nat := row8521_block_009 ++ row8521_block_010
theorem row8521_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8521 row8521_large_right_left :=
  Budget.RowUpperBound.branch row8521_block_009_covers row8521_block_010_covers

def row8521_large_right_right_right : List Nat := row8521_block_012 ++ row8521_block_013
theorem row8521_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8521 row8521_large_right_right_right :=
  Budget.RowUpperBound.branch row8521_block_012_covers row8521_block_013_covers

def row8521_large_right_right : List Nat := row8521_block_011 ++ row8521_large_right_right_right
theorem row8521_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8521 row8521_large_right_right :=
  Budget.RowUpperBound.branch row8521_block_011_covers row8521_large_right_right_right_covers

def row8521_large_right : List Nat := row8521_large_right_left ++ row8521_large_right_right
theorem row8521_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8521 row8521_large_right :=
  Budget.RowUpperBound.branch row8521_large_right_left_covers row8521_large_right_right_covers

def row8521_large : List Nat := row8521_large_left ++ row8521_large_right
theorem row8521_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8521 row8521_large :=
  Budget.RowUpperBound.branch row8521_large_left_covers row8521_large_right_covers

def row8521 : List Nat := row8521_large
theorem row8521_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8521 row8521 := row8521_large_covers

end ProjectEulerSolutions.PrimePairLarge
