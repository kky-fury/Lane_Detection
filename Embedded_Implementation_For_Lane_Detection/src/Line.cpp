#include"line.hpp"


Line::Line(Linepoint startpoint, Linepoint endpoint)
{
	this->startpoint.x = startpoint.x;
	this->startpoint.y = endpoint.y;
	
	this->endpoint.x  = endpoint.x;
	this->endpoint.y = endpoint.y;


}

Linepoint Line:: getstartpoint()
{
	return this->startpoint;
	
}

Linepoint Line::getendpoint()
{

	return this->endpoint;
	
}
