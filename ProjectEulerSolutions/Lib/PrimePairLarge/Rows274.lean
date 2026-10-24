import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row8821_small : List Nat := []
theorem row8821_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8821 row8821_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8821_block_005 : List Nat := [9067, 9319, 9883]
def plan8821_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 31), .first, .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 13), (.firstFactor 23), .first, .second, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .possible, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .second, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .first, .possible, .first, (.firstFactor 19), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, .first, (.firstFactor 23), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 19), .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, .second, .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 23), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8821_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8821 row8821_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8821_block_005
  · decide
  · rfl

def row8821_block_006 : List Nat := [10459, 10861, 11173, 12043]
def plan8821_block_006 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .possible, (.firstFactor 11), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 17), .first, .possible, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 29), (.secondFactor 29), (.firstFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .possible, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first]
set_option maxRecDepth 2048 in
theorem row8821_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8821 row8821_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8821_block_006
  · decide
  · rfl

def row8821_block_007 : List Nat := [13381, 14281]
def plan8821_block_007 : List Budget.Choice := [.second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 17), (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, .second, (.firstFactor 13), (.secondFactor 23), .second, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 31), .first, .possible, .first, (.firstFactor 19), (.firstFactor 23), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 19), (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 23), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.secondFactor 17), .first, .first, .first, .possible, .second, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .second, .first]
set_option maxRecDepth 2048 in
theorem row8821_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8821 row8821_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8821_block_007
  · decide
  · rfl

def row8821_block_008 : List Nat := [15319, 15607]
def plan8821_block_008 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 11), .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .possible, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 29), (.secondFactor 31), (.secondFactor 23), .first, .first, .first, (.secondFactor 13), (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8821_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8821 row8821_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8821_block_008
  · decide
  · rfl

def row8821_block_009 : List Nat := [16561]
def plan8821_block_009 : List Budget.Choice := [(.firstFactor 17), .first, .first, .first, .possible, (.secondFactor 7), (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 11), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 11), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 29), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8821_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8821 row8821_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8821_block_009
  · decide
  · rfl

def row8821_block_010 : List Nat := []
theorem row8821_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8821 row8821_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8821_block_011 : List Nat := []
theorem row8821_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8821 row8821_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8821_block_012 : List Nat := []
theorem row8821_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8821 row8821_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8821_block_013 : List Nat := []
theorem row8821_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8821 row8821_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8821_large_left_left : List Nat := row8821_small ++ row8821_block_005
theorem row8821_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8821 row8821_large_left_left :=
  Budget.RowUpperBound.branch row8821_small_covers row8821_block_005_covers

def row8821_large_left_right_right : List Nat := row8821_block_007 ++ row8821_block_008
theorem row8821_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8821 row8821_large_left_right_right :=
  Budget.RowUpperBound.branch row8821_block_007_covers row8821_block_008_covers

def row8821_large_left_right : List Nat := row8821_block_006 ++ row8821_large_left_right_right
theorem row8821_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8821 row8821_large_left_right :=
  Budget.RowUpperBound.branch row8821_block_006_covers row8821_large_left_right_right_covers

def row8821_large_left : List Nat := row8821_large_left_left ++ row8821_large_left_right
theorem row8821_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8821 row8821_large_left :=
  Budget.RowUpperBound.branch row8821_large_left_left_covers row8821_large_left_right_covers

def row8821_large_right_left : List Nat := row8821_block_009 ++ row8821_block_010
theorem row8821_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8821 row8821_large_right_left :=
  Budget.RowUpperBound.branch row8821_block_009_covers row8821_block_010_covers

def row8821_large_right_right_right : List Nat := row8821_block_012 ++ row8821_block_013
theorem row8821_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8821 row8821_large_right_right_right :=
  Budget.RowUpperBound.branch row8821_block_012_covers row8821_block_013_covers

def row8821_large_right_right : List Nat := row8821_block_011 ++ row8821_large_right_right_right
theorem row8821_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8821 row8821_large_right_right :=
  Budget.RowUpperBound.branch row8821_block_011_covers row8821_large_right_right_right_covers

def row8821_large_right : List Nat := row8821_large_right_left ++ row8821_large_right_right
theorem row8821_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8821 row8821_large_right :=
  Budget.RowUpperBound.branch row8821_large_right_left_covers row8821_large_right_right_covers

def row8821_large : List Nat := row8821_large_left ++ row8821_large_right
theorem row8821_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8821 row8821_large :=
  Budget.RowUpperBound.branch row8821_large_left_covers row8821_large_right_covers

def row8821 : List Nat := row8821_large
theorem row8821_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8821 row8821 := row8821_large_covers

def row8831_small : List Nat := []
theorem row8831_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8831 row8831_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8831_block_005 : List Nat := [8837, 9293, 10301]
def plan8831_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .possible, .first, (.secondFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 31), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 31), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .possible, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.firstFactor 37), (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .second, .first, .second, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .possible, .first, (.secondFactor 23), .first, (.secondFactor 7), .first, (.secondFactor 31), (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row8831_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8831 row8831_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8831_block_005
  · decide
  · rfl

def row8831_block_006 : List Nat := [11399, 12203]
def plan8831_block_006 : List Budget.Choice := [.second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 17), .second, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 11), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .possible, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 11), (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .second, .first, (.secondFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.secondFactor 13), .second, .second, .first, .first, .first, (.firstFactor 11), .possible, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), .first, (.secondFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row8831_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8831 row8831_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8831_block_006
  · decide
  · rfl

def row8831_block_007 : List Nat := [13109, 14009]
def plan8831_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .possible, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 17), (.firstFactor 31), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .second, .second, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 11), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8831_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8831 row8831_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8831_block_007
  · decide
  · rfl

def row8831_block_008 : List Nat := [16061]
def plan8831_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, (.secondFactor 11), .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.firstFactor 13), .first, (.secondFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .second, (.firstFactor 11), .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .second, .first, (.secondFactor 7), (.firstFactor 23), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .possible, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 13), (.secondFactor 31), .first, (.firstFactor 31), (.secondFactor 11), .first, .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .second, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row8831_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8831 row8831_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8831_block_008
  · decide
  · rfl

def row8831_block_009 : List Nat := []
def plan8831_block_009 : List Budget.Choice := [.first, .second, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 17), (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .second, (.firstFactor 17), .first, .first, .first, (.secondFactor 11), (.secondFactor 23), .first, .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8831_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8831 row8831_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8831_block_009
  · decide
  · rfl

def row8831_block_010 : List Nat := []
theorem row8831_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8831 row8831_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8831_block_011 : List Nat := []
theorem row8831_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8831 row8831_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8831_block_012 : List Nat := []
theorem row8831_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8831 row8831_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8831_block_013 : List Nat := []
theorem row8831_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8831 row8831_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8831_large_left_left : List Nat := row8831_small ++ row8831_block_005
theorem row8831_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8831 row8831_large_left_left :=
  Budget.RowUpperBound.branch row8831_small_covers row8831_block_005_covers

def row8831_large_left_right_right : List Nat := row8831_block_007 ++ row8831_block_008
theorem row8831_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8831 row8831_large_left_right_right :=
  Budget.RowUpperBound.branch row8831_block_007_covers row8831_block_008_covers

def row8831_large_left_right : List Nat := row8831_block_006 ++ row8831_large_left_right_right
theorem row8831_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8831 row8831_large_left_right :=
  Budget.RowUpperBound.branch row8831_block_006_covers row8831_large_left_right_right_covers

def row8831_large_left : List Nat := row8831_large_left_left ++ row8831_large_left_right
theorem row8831_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8831 row8831_large_left :=
  Budget.RowUpperBound.branch row8831_large_left_left_covers row8831_large_left_right_covers

def row8831_large_right_left : List Nat := row8831_block_009 ++ row8831_block_010
theorem row8831_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8831 row8831_large_right_left :=
  Budget.RowUpperBound.branch row8831_block_009_covers row8831_block_010_covers

def row8831_large_right_right_right : List Nat := row8831_block_012 ++ row8831_block_013
theorem row8831_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8831 row8831_large_right_right_right :=
  Budget.RowUpperBound.branch row8831_block_012_covers row8831_block_013_covers

def row8831_large_right_right : List Nat := row8831_block_011 ++ row8831_large_right_right_right
theorem row8831_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8831 row8831_large_right_right :=
  Budget.RowUpperBound.branch row8831_block_011_covers row8831_large_right_right_right_covers

def row8831_large_right : List Nat := row8831_large_right_left ++ row8831_large_right_right
theorem row8831_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8831 row8831_large_right :=
  Budget.RowUpperBound.branch row8831_large_right_left_covers row8831_large_right_right_covers

def row8831_large : List Nat := row8831_large_left ++ row8831_large_right
theorem row8831_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8831 row8831_large :=
  Budget.RowUpperBound.branch row8831_large_left_covers row8831_large_right_covers

def row8831 : List Nat := row8831_large
theorem row8831_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8831 row8831 := row8831_large_covers

def row8837_small : List Nat := []
theorem row8837_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8837 row8837_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8837_block_005 : List Nat := [8951, 9371, 9623]
def plan8837_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 29), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .possible, (.secondFactor 29), .first, .first, .first, (.firstFactor 17), (.firstFactor 31), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 7), (.secondFactor 13), .second, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .second, .first, .first, .first, .possible, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 11), .first, (.firstFactor 7), .second, (.secondFactor 11), .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row8837_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8837 row8837_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8837_block_005
  · decide
  · rfl

def row8837_block_006 : List Nat := [10601, 11399, 11483, 11657, 12197]
def plan8837_block_006 : List Budget.Choice := [(.secondFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 11), .first, .possible, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 19), (.firstFactor 11), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .possible, (.firstFactor 19), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .possible, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.secondFactor 29), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 31), .second, (.secondFactor 11), .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .second, .first, (.secondFactor 13), .first, .first, (.secondFactor 29), .first, .possible, (.firstFactor 11), .first, .second, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 19)]
set_option maxRecDepth 2048 in
theorem row8837_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8837 row8837_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8837_block_006
  · decide
  · rfl

def row8837_block_007 : List Nat := [12491, 13463]
def plan8837_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .second, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 29), .first, .first, (.firstFactor 23), .first, .second, (.secondFactor 17), .first, .first, .first, (.secondFactor 19), (.firstFactor 31), .first, (.secondFactor 7), .first, .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.firstFactor 11), .possible, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), (.secondFactor 11), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 17), .second, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row8837_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8837 row8837_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8837_block_007
  · decide
  · rfl

def row8837_block_008 : List Nat := [16187]
def plan8837_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, .first, .second, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 17), .first, .first, (.secondFactor 13), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 29), (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, (.secondFactor 11), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 37), .first, (.firstFactor 7), .second, .first, .first, .possible, .first, (.secondFactor 23), (.firstFactor 29), (.firstFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row8837_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8837 row8837_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8837_block_008
  · decide
  · rfl

def row8837_block_009 : List Nat := [16811, 17117]
def plan8837_block_009 : List Budget.Choice := [.first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), .possible, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 13), .first, .first, .first, .second, (.secondFactor 17), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .possible, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8837_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8837 row8837_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8837_block_009
  · decide
  · rfl

def row8837_block_010 : List Nat := []
theorem row8837_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8837 row8837_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8837_block_011 : List Nat := []
theorem row8837_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8837 row8837_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8837_block_012 : List Nat := []
theorem row8837_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8837 row8837_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8837_block_013 : List Nat := []
theorem row8837_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8837 row8837_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8837_large_left_left : List Nat := row8837_small ++ row8837_block_005
theorem row8837_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8837 row8837_large_left_left :=
  Budget.RowUpperBound.branch row8837_small_covers row8837_block_005_covers

def row8837_large_left_right_right : List Nat := row8837_block_007 ++ row8837_block_008
theorem row8837_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8837 row8837_large_left_right_right :=
  Budget.RowUpperBound.branch row8837_block_007_covers row8837_block_008_covers

def row8837_large_left_right : List Nat := row8837_block_006 ++ row8837_large_left_right_right
theorem row8837_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8837 row8837_large_left_right :=
  Budget.RowUpperBound.branch row8837_block_006_covers row8837_large_left_right_right_covers

def row8837_large_left : List Nat := row8837_large_left_left ++ row8837_large_left_right
theorem row8837_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8837 row8837_large_left :=
  Budget.RowUpperBound.branch row8837_large_left_left_covers row8837_large_left_right_covers

def row8837_large_right_left : List Nat := row8837_block_009 ++ row8837_block_010
theorem row8837_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8837 row8837_large_right_left :=
  Budget.RowUpperBound.branch row8837_block_009_covers row8837_block_010_covers

def row8837_large_right_right_right : List Nat := row8837_block_012 ++ row8837_block_013
theorem row8837_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8837 row8837_large_right_right_right :=
  Budget.RowUpperBound.branch row8837_block_012_covers row8837_block_013_covers

def row8837_large_right_right : List Nat := row8837_block_011 ++ row8837_large_right_right_right
theorem row8837_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8837 row8837_large_right_right :=
  Budget.RowUpperBound.branch row8837_block_011_covers row8837_large_right_right_right_covers

def row8837_large_right : List Nat := row8837_large_right_left ++ row8837_large_right_right
theorem row8837_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8837 row8837_large_right :=
  Budget.RowUpperBound.branch row8837_large_right_left_covers row8837_large_right_right_covers

def row8837_large : List Nat := row8837_large_left ++ row8837_large_right
theorem row8837_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8837 row8837_large :=
  Budget.RowUpperBound.branch row8837_large_left_covers row8837_large_right_covers

def row8837 : List Nat := row8837_large
theorem row8837_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8837 row8837 := row8837_large_covers

def row8839_small : List Nat := []
theorem row8839_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 8839 row8839_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row8839_block_005 : List Nat := []
def plan8839_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 11), (.secondFactor 19), (.firstFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.secondFactor 31), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 19), (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 17), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 37), .first, (.secondFactor 19), (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 11), .first, .second, (.secondFactor 17), .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row8839_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 8839 row8839_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan8839_block_005
  · decide
  · rfl

def row8839_block_006 : List Nat := [10429, 11047, 11971]
def plan8839_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, .possible, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 37), (.secondFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .second, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .possible, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 23), (.firstFactor 17), (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .second, .first, .first, .first, .first, (.secondFactor 17), .second, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 23), .first, (.firstFactor 13), (.secondFactor 29), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .second, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 13), .second, .first, .first, (.secondFactor 19), .first, (.firstFactor 29), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8839_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 8839 row8839_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan8839_block_006
  · decide
  · rfl

def row8839_block_007 : List Nat := [12829, 12979, 13441]
def plan8839_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 17), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .second, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 11), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 29), .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, .possible, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .first, .second, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 13), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 19), (.secondFactor 17), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 29)]
set_option maxRecDepth 2048 in
theorem row8839_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 8839 row8839_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan8839_block_007
  · decide
  · rfl

def row8839_block_008 : List Nat := [14653, 14929]
def plan8839_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 13), (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 13), .second, (.firstFactor 37), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 29), .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 11), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .second, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 23), (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row8839_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 8839 row8839_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan8839_block_008
  · decide
  · rfl

def row8839_block_009 : List Nat := [16603]
def plan8839_block_009 : List Budget.Choice := [(.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .possible, .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 19), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 13), (.firstFactor 19), (.secondFactor 11), .first, .first, .second, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row8839_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 8839 row8839_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan8839_block_009
  · decide
  · rfl

def row8839_block_010 : List Nat := []
theorem row8839_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 8839 row8839_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row8839_block_011 : List Nat := []
theorem row8839_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 8839 row8839_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row8839_block_012 : List Nat := []
theorem row8839_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 8839 row8839_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row8839_block_013 : List Nat := []
theorem row8839_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 8839 row8839_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row8839_large_left_left : List Nat := row8839_small ++ row8839_block_005
theorem row8839_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 8839 row8839_large_left_left :=
  Budget.RowUpperBound.branch row8839_small_covers row8839_block_005_covers

def row8839_large_left_right_right : List Nat := row8839_block_007 ++ row8839_block_008
theorem row8839_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 8839 row8839_large_left_right_right :=
  Budget.RowUpperBound.branch row8839_block_007_covers row8839_block_008_covers

def row8839_large_left_right : List Nat := row8839_block_006 ++ row8839_large_left_right_right
theorem row8839_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 8839 row8839_large_left_right :=
  Budget.RowUpperBound.branch row8839_block_006_covers row8839_large_left_right_right_covers

def row8839_large_left : List Nat := row8839_large_left_left ++ row8839_large_left_right
theorem row8839_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 8839 row8839_large_left :=
  Budget.RowUpperBound.branch row8839_large_left_left_covers row8839_large_left_right_covers

def row8839_large_right_left : List Nat := row8839_block_009 ++ row8839_block_010
theorem row8839_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 8839 row8839_large_right_left :=
  Budget.RowUpperBound.branch row8839_block_009_covers row8839_block_010_covers

def row8839_large_right_right_right : List Nat := row8839_block_012 ++ row8839_block_013
theorem row8839_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 8839 row8839_large_right_right_right :=
  Budget.RowUpperBound.branch row8839_block_012_covers row8839_block_013_covers

def row8839_large_right_right : List Nat := row8839_block_011 ++ row8839_large_right_right_right
theorem row8839_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 8839 row8839_large_right_right :=
  Budget.RowUpperBound.branch row8839_block_011_covers row8839_large_right_right_right_covers

def row8839_large_right : List Nat := row8839_large_right_left ++ row8839_large_right_right
theorem row8839_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 8839 row8839_large_right :=
  Budget.RowUpperBound.branch row8839_large_right_left_covers row8839_large_right_right_covers

def row8839_large : List Nat := row8839_large_left ++ row8839_large_right
theorem row8839_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 8839 row8839_large :=
  Budget.RowUpperBound.branch row8839_large_left_covers row8839_large_right_covers

def row8839 : List Nat := row8839_large
theorem row8839_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 8839 row8839 := row8839_large_covers

end ProjectEulerSolutions.PrimePairLarge
