require 'ply'

pf = Ply::PlyFile.new"FF.ply"
puts pf.data.keys

puts pf.data["vertex"][0]