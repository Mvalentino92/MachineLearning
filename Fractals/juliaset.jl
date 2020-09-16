using IterTools
using Dates
using Plots
using Images, ImageDraw, ImageMagick
using FileIO, CSV
using DataFrames

# Negating tuples
neg(T) = map(x -> -x,T)

function inset(z::Complex,c::Complex,R::Real,maxiter::Int)
	for i = 1:maxiter
		z = z*z + c
		if abs(z) > R
			return false
		end
	end
	return true
end

function J(c::Complex;maxdepth::Int=2,h::Real=0.01073937,maxiter::Int=35,set::Vector=[])
	
	# Find R
	R = max(abs(c),2)

	# If not passing initial set of points to recur expand, generate
	if isempty(set)

		# Generate bound
		bound = sqrt(R)

		# Create intial ranges
		xrange = 0:h:bound
		yrange = -bound:h:bound

		# Sort the yrange, so we can cut out circle of radius R
		yrange = sort(yrange,by=(x -> abs(x)))

		# Create the set of initial points (speed up by negating any found)
		set = [(x,y) for x in xrange for y in 
		      takewhile(y -> x*x + y*y <= R,yrange) 
		      if inset(x + (y)im,c,R,maxiter)]

		# Grab negatives
		set = merge(set,map(x -> neg(x),set))
	end

	# Begin recursive calls for generating tighter and tighter points
	return isempty(set) ? set : 
	append!(set,mapreduce(x -> exploit(x,c,R,h/2,maxdepth,maxiter),(a,b) -> append!(a,b),set))
end

function exploit(z::Tuple,c::Complex,R::Real,h::Real,depth::Int,maxiter::Int)
	
	# If depth = 0 return empty
	if depth == 0 return [] end

	# Get new ranges
	xs,xt,ys,yt = z[1]-h*0.95,z[1]+h*0.95,z[2]-h*0.95,z[2]+h*0.95
	xrange = LinRange(xs,xt,4)
	yrange = LinRange(ys,yt,4)
	h = ((xrange[2] - xrange[1]) + (yrange[2]-yrange[1]))*0.5

	# Generate set
	set = [(x,y) for x in xrange for y in yrange if inset(x + (y)im,c,R,maxiter)]

	# Begin recursive calls for generating tighter and tighter points
	return isempty(set) ? set : 
	append!(set,mapreduce(x -> exploit(x,c,R,h/2,depth-1,maxiter),(a,b) -> append!(a,b),set))
end

# Go from real number to pixel
function real_to_pixel(r,mn_r,mx_r,mn_p,mx_p)
	return round(Int64,mn_p + (r-mn_r)/(mx_r - mn_r)*(mx_p - mn_p))
end

# Turn the set of points in real numbers, to pixels (flip because julia is column based)
function points_to_pixels(ps,mn_r,mx_r,mn_p,mx_p)
	return map(p -> Point(real_to_pixel(p[1],mn_r,mx_r,mn_p,mx_p),real_to_pixel(p[2],mn_r,mx_r,mn_p,mx_p)),ps)
end

# Generate fractal with specific c
function frac(c::Complex;filename::String="tester.jpeg",crange::Real=sqrt(2),mn_p::Int=1,mx_p::Int=400)
	set = J(c,maxdepth=0)
	set = points_to_pixels(J(c,set=set),-crange,crange,mn_p,mx_p)
	img = ones(Gray,mx_p,mx_p)
	draw!(img,set,Gray(colorant"black"))
	save(filename,img)
end

# Generate fractals
function fractals(num::Int;crange::Real=sqrt(2),mn_p::Int=1,mx_p::Int=400)
	dte = string(pwd(),"/Fractals_",today(),"/")
	if !isdir(dte) mkdir(dte) end
	realpart = zeros(Real,num)
	imagpart = zeros(Real,num)
	total = num
	while num > 0
		c = 2*crange*(rand()-0.5) + (2*crange*(rand()-0.5))im
		initset = J(c,maxdepth=0)
		l = length(initset)
		if l < 17500 && l > 1000
			img = ones(Gray,mx_p,mx_p)
			set = points_to_pixels(J(c,set=initset),-crange,crange,mn_p,mx_p)
			draw!(img,set,Gray(colorant"black"))
			num -= 1
			idx = total - num
			save(string(dte,"FRAC_",(idx-1+2100),".jpeg"),img)
			realpart[idx] = real(c)
			imagpart[idx] = imag(c)
			println(idx)
		end
	end
	df = DataFrame(Real = realpart, Imaginary = imagpart)
	CSV.write(string(dte,"cvalues_2100.csv"),df)
end

function mandelbrot(;bound::Real=pi,h::Real=0.01073937,maxiter::Int=35)

	# Generate the set by pure testing
	range = -bound:h:bound
	return [(x,y) for x in range for y in range if inset(x + (y)im, x + (y)im, max(2,sqrt(x*x + y*y)),35)]
end

